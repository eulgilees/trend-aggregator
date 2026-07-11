#!/usr/bin/env python3
import json
import os
import re
import sqlite3
import threading
import time
import urllib.request
from calendar import timegm
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs

import feedparser

FULL_CONTENT_THRESHOLD = 400  # 이 길이를 넘으면 "전체 내용"으로 간주

DB_PATH = 'trend.db'
FEEDS_PATH = 'feeds.json'
REFRESH_INTERVAL = 15 * 60  # 15분마다 전체 피드 재수집
PORT = int(os.environ.get('PORT', 8000))

ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY', '')
ANTHROPIC_MODEL = 'claude-haiku-4-5-20251001'
AI_SUMMARY_BATCH = 20  # 새로고침 1회당 최대 요약 생성 건수 (비용/속도 제어)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def load_feeds():
    with open(FEEDS_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            category TEXT NOT NULL,
            title TEXT NOT NULL,
            link TEXT NOT NULL UNIQUE,
            summary TEXT,
            content_html TEXT,
            is_full INTEGER NOT NULL DEFAULT 0,
            ai_summary TEXT,
            ai_summary_at INTEGER,
            featured INTEGER NOT NULL DEFAULT 0,
            feature_reason TEXT,
            published_ts INTEGER,
            fetched_at INTEGER NOT NULL
        )
    ''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_items_published ON items(published_ts DESC)')
    existing_cols = {row[1] for row in conn.execute('PRAGMA table_info(items)').fetchall()}
    for col, coltype in (('featured', 'INTEGER NOT NULL DEFAULT 0'), ('feature_reason', 'TEXT')):
        if col not in existing_cols:
            conn.execute(f'ALTER TABLE items ADD COLUMN {col} {coltype}')
    conn.commit()
    conn.close()


def entry_timestamp(entry):
    for key in ('published_parsed', 'updated_parsed'):
        val = entry.get(key)
        if val:
            return timegm(val)
    return int(time.time())


def strip_html(text, limit=None):
    text = re.sub(r'<[^>]+>', ' ', text or '')
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:limit] if limit else text


def sanitize_html(html):
    """RSS/이메일에서 온 HTML을 그대로 렌더링하기 전 위험 요소 제거."""
    html = html or ''
    html = re.sub(r'(?is)<script.*?>.*?</script>', '', html)
    html = re.sub(r'(?is)<style.*?>.*?</style>', '', html)
    html = re.sub(r'(?is)<iframe.*?>.*?</iframe>', '', html)
    html = re.sub(r'(?i)\son\w+\s*=\s*(".*?"|\'.*?\')', '', html)
    html = re.sub(r'(?i)(href|src)\s*=\s*(["\'])\s*javascript:.*?\2', r'\1=\2#\2', html)
    return html


def entry_full_content(entry):
    if entry.get('content'):
        return entry['content'][0].get('value', '') or ''
    return ''


def fetch_feed(source):
    parsed = feedparser.parse(source['url'])
    rows = []
    for entry in parsed.entries:
        link = entry.get('link') or entry.get('id')
        title = entry.get('title')
        if not link or not title:
            continue
        raw_summary = entry.get('summary', '') or ''
        raw_content = entry_full_content(entry)
        best_html = raw_content if len(strip_html(raw_content)) > len(strip_html(raw_summary)) else raw_summary
        is_full = 1 if len(strip_html(best_html)) > FULL_CONTENT_THRESHOLD else 0
        rows.append((
            source['name'],
            source['category'],
            title,
            link,
            strip_html(raw_summary, limit=200),
            sanitize_html(best_html),
            is_full,
            entry_timestamp(entry),
            int(time.time()),
        ))
    return rows


def extract_article_text(url, limit=8000):
    """RSS가 발췌문만 줄 때, 원문 페이지에서 본문 텍스트를 최대한 뽑아온다."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (trend-aggregator)'})
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
    except Exception as e:
        print(f'[extract] {url} 실패: {e}')
        return None
    html = re.sub(r'(?is)<(script|style|nav|header|footer|aside|noscript|form).*?>.*?</\1>', '', html)
    return strip_html(html, limit=limit)


def trim_to_last_sentence(text):
    """max_tokens에 걸려 잘린 경우, 문장 중간에서 끝나지 않도록 마지막 완결 문장까지만 남긴다."""
    matches = list(re.finditer(r'[.!?다요음]\s', text))
    if not matches:
        return text
    end = matches[-1].end()
    return text[:end].strip()


def summarize_with_claude(text, title):
    if not ANTHROPIC_API_KEY or not text:
        return None
    text = text[:6000]
    prompt = (
        f"다음은 '{title}'라는 글의 내용입니다. 글을 안 읽어도 핵심을 바로 알 수 있게 "
        f"한국어로 2~3문장으로 짧고 완결된 요약을 해줘. 요약문만 출력하고 다른 말은 붙이지 마.\n\n{text}"
    )
    body = json.dumps({
        'model': ANTHROPIC_MODEL,
        'max_tokens': 500,
        'messages': [{'role': 'user', 'content': prompt}],
    }).encode('utf-8')
    req = urllib.request.Request(
        'https://api.anthropic.com/v1/messages',
        data=body,
        headers={
            'x-api-key': ANTHROPIC_API_KEY,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        summary = data['content'][0]['text'].strip()
        if data.get('stop_reason') == 'max_tokens':
            summary = trim_to_last_sentence(summary)
        return summary
    except Exception as e:
        print(f'[summarize] 실패: {e}')
        return None


def backfill_ai_summaries(conn):
    if not ANTHROPIC_API_KEY:
        return
    rows = conn.execute('''
        SELECT id, link, title, content_html, is_full, summary FROM items
        WHERE ai_summary IS NULL
        ORDER BY published_ts DESC LIMIT ?
    ''', (AI_SUMMARY_BATCH,)).fetchall()
    for item_id, link, title, content_html, is_full, summary in rows:
        source_text = strip_html(content_html) if is_full and content_html else None
        if not source_text:
            source_text = extract_article_text(link) or summary or ''
        ai_summary = summarize_with_claude(source_text, title)
        if ai_summary:
            conn.execute(
                'UPDATE items SET ai_summary=?, ai_summary_at=? WHERE id=?',
                (ai_summary, int(time.time()), item_id),
            )
            conn.commit()
            print(f'[ai] 요약 완료: {title[:30]}')


def select_featured_items(conn):
    if not ANTHROPIC_API_KEY:
        return
    rows = conn.execute('''
        SELECT id, source, title, COALESCE(ai_summary, summary, '') FROM items
        ORDER BY published_ts DESC LIMIT 40
    ''').fetchall()
    if not rows:
        return
    listing = '\n'.join(
        f"{i + 1}. [{r[1]}] {r[2]} - {strip_html(r[3])[:120]}" for i, r in enumerate(rows)
    )
    prompt = (
        "다음은 최근 수집된 트렌드 뉴스 목록이야. 문구/라이프스타일 굿즈를 파는 매장(핫트랙스 같은 곳)의 "
        "상품 기획·머천다이징에 참고할 만큼 최근 가장 유행하고 있고 인사이트를 줄 만한 항목을 최대 4개 골라줘. "
        "다른 설명 없이 JSON만 출력해: {\"picks\": [{\"index\": 번호, \"reason\": \"한 줄 이유\"}]}\n\n"
        f"{listing}"
    )
    body = json.dumps({
        'model': ANTHROPIC_MODEL,
        'max_tokens': 500,
        'messages': [{'role': 'user', 'content': prompt}],
    }).encode('utf-8')
    req = urllib.request.Request(
        'https://api.anthropic.com/v1/messages',
        data=body,
        headers={
            'x-api-key': ANTHROPIC_API_KEY,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode('utf-8'))
        text = data['content'][0]['text'].strip()
        match = re.search(r'\{.*\}', text, re.S)
        if not match:
            print(f'[feature] 파싱 실패: JSON 없음 -> {text[:200]!r}')
            return
        picks = json.loads(match.group(0)).get('picks', [])
    except Exception as e:
        print(f'[feature] 실패: {e}')
        return

    conn.execute('UPDATE items SET featured=0, feature_reason=NULL')
    picked = 0
    for p in picks:
        idx = p.get('index')
        reason = p.get('reason', '')
        if not isinstance(idx, int) or not (1 <= idx <= len(rows)):
            continue
        item_id = rows[idx - 1][0]
        conn.execute('UPDATE items SET featured=1, feature_reason=? WHERE id=?', (reason, item_id))
        picked += 1
    conn.commit()
    print(f'[feature] {picked}건 선정')


def refresh_all():
    feeds = load_feeds()
    conn = sqlite3.connect(DB_PATH)
    for source in feeds:
        try:
            rows = fetch_feed(source)
        except Exception as e:
            print(f'[refresh] {source["name"]} 실패: {e}')
            continue
        conn.executemany('''
            INSERT INTO items (source, category, title, link, summary, content_html, is_full, published_ts, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(link) DO NOTHING
        ''', rows)
        conn.commit()
        print(f'[refresh] {source["name"]}: {len(rows)}건 확인')
    backfill_ai_summaries(conn)
    select_featured_items(conn)
    conn.close()


def refresh_loop():
    while True:
        refresh_all()
        time.sleep(REFRESH_INTERVAL)


def query_items(category=None, source=None, limit=100):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    sql = '''SELECT source, category, title, link, summary, content_html, is_full, ai_summary,
                     published_ts, featured, feature_reason FROM items'''
    conditions, params = [], []
    if category:
        conditions.append('category = ?')
        params.append(category)
    if source:
        conditions.append('source = ?')
        params.append(source)
    if conditions:
        sql += ' WHERE ' + ' AND '.join(conditions)
    sql += ' ORDER BY published_ts DESC LIMIT ?'
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path, content_type):
        try:
            with open(path, 'rb') as f:
                body = f.read()
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path == '/':
            self._send_file('index.html', 'text/html; charset=utf-8')
        elif parsed.path == '/api/items':
            items = query_items(
                category=qs.get('category', [None])[0],
                source=qs.get('source', [None])[0],
                limit=int(qs.get('limit', [100])[0]),
            )
            self._send_json({'items': items})
        elif parsed.path == '/api/sources':
            self._send_json({'sources': load_feeds()})
        else:
            self.send_response(404)
            self.end_headers()


if __name__ == '__main__':
    init_db()
    threading.Thread(target=refresh_loop, daemon=True).start()
    server = ThreadedHTTPServer(('0.0.0.0', PORT), Handler)
    print(f'서버 실행 중: http://0.0.0.0:{PORT}')
    server.serve_forever()
