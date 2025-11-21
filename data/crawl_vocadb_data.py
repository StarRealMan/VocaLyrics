#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
按排名爬取 VocaDB Original 歌曲，保存必要 metadata。
字段：

    id
    defaultName
    year
    ratingScore
    favoritedTimes
    lengthSeconds
    artists: [
        { id, name, role }  # role 由 categories / effectiveRoles 推导
    ]
    tags: [
        { tagId, tagName, count }
    ]
    producerNames: [ ... ]
    primaryCultureCode
    originalLyrics
    mainPictureUrlOriginal

限速参数：
    --delay         每次请求之间的基础延时
    --rest-every    每下载多少首歌额外休息一次
    --rest-seconds  额外休息时长
"""

import argparse
import json
import logging
import re
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

BASE_URL = "https://vocadb.net"
SONGS_ENDPOINT = f"{BASE_URL}/api/songs"

DETAIL_FIELDS = ",".join([
    "Artists",
    "Albums",
    "Tags",
    "Lyrics",
    "MainPicture",
])

USER_AGENT = "VocaDB-Crawler (For Academic Research) (Contact: starydyxyz@gmail.com)"


def setup_logger() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=0, help="爬取前 N 名歌曲")
    parser.add_argument("--output_dir", type=str, default="./test", help="输出目录")
    parser.add_argument("--page_size", type=int, default=100, help="排行榜每页条数")
    parser.add_argument("--delay", type=float, default=1.0, help="每次请求前延时秒数")
    parser.add_argument("--max_retries", type=int, default=5, help="失败重试次数")
    parser.add_argument("--rest_every", type=int, default=0, help="每下载多少首歌额外休眠一次（0=不开）")
    parser.add_argument("--rest_seconds", type=float, default=60.0, help="额外休眠秒数")
    return parser.parse_args()


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def safe_get_json(
    session: requests.Session,
    url: str,
    params: Dict[str, Any],
    delay: float,
    max_retries: int,
) -> Optional[Dict[str, Any]]:
    """带限速和重试的 GET JSON。"""
    for attempt in range(max_retries):
        if delay > 0:
            time.sleep(delay)

        try:
            resp = session.get(url, params=params, timeout=20)
        except requests.RequestException as e:
            logging.warning("请求异常：%s", e)
            wait = delay * (2 ** attempt) + random.uniform(0, delay)
            logging.info("等待 %.2f 秒后重试（第 %d 次）", wait, attempt + 1)
            time.sleep(wait)
            continue

        if resp.status_code == 200:
            try:
                return resp.json()
            except Exception as e:
                logging.warning("JSON 解析失败：%s", e)
                return None

        if resp.status_code in (429, 500, 502, 503, 504):
            wait = delay * (2 ** attempt) + random.uniform(0, delay)
            logging.warning("HTTP %d，等待 %.2f 秒后重试（第 %d 次）",
                            resp.status_code, wait, attempt + 1)
            time.sleep(wait)
            continue

        logging.error("HTTP %d 错误，停止：url=%s params=%s", resp.status_code, url, params)
        return None

    logging.error("超过最大重试次数：url=%s params=%s", url, params)
    return None


def load_existing_state(output_dir: Path) -> Tuple[Set[int], int]:
    """
    扫描 output_dir 下的 song_*.json：
      - 收集已存在的歌曲 id 集合
      - 找到已完成的最大 rank（用于断点续传）
    """
    existing_ids: Set[int] = set()
    max_rank = 0

    if not output_dir.exists():
        return existing_ids, max_rank

    pattern = re.compile(r"song_(\d+)\.json$")
    for path in output_dir.glob("song_*.json"):
        m = pattern.match(path.name)
        if not m:
            continue
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        song_id = data.get("id")
        if isinstance(song_id, int):
            existing_ids.add(song_id)

        rank = data.get("rank")
        if isinstance(rank, int) and rank > max_rank:
            max_rank = rank

    logging.info("已有歌曲：%d，已完成最大 rank=%d", len(existing_ids), max_rank)
    return existing_ids, max_rank


def choose_original_lyrics(song_detail: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从 Lyrics 列表中选出 translationType == 'Original' 的一条。"""
    lyrics_list = song_detail.get("lyrics") or song_detail.get("Lyrics") or []
    if not isinstance(lyrics_list, list):
        return None
    for lyr in lyrics_list:
        if isinstance(lyr, dict) and lyr.get("translationType") == "Original":
            return lyr
    return None


def extract_year(date_str: Optional[str]) -> Optional[int]:
    if not date_str or len(date_str) < 4:
        return None
    try:
        return int(date_str[:4])
    except Exception:
        return None

def extract_month(date_str: Optional[str]) -> Optional[int]:
    if not date_str or len(date_str) < 7:
        return None
    try:
        return int(date_str[5:7])
    except Exception:
        return None

def simplify_artists(artists_raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将 VocaDB 的 artists 列表简化为：
        { id, name, role }
    role 计算规则：
        - role := categories
        - 若 categories == "Other"，则 role := effectiveRoles
    id/name 优先使用内层 artist.id / artist.name。
    """
    simplified: List[Dict[str, Any]] = []

    for a in artists_raw:
        if not isinstance(a, dict):
            continue

        inner = a.get("artist") or {}
        if not isinstance(inner, dict):
            inner = {}

        artist_id = inner.get("id") or a.get("id")
        artist_name = inner.get("name") or a.get("name")

        role = a.get("categories")
        if role == "Other" or not role:
            role = a.get("effectiveRoles")

        simplified.append({
            "id": artist_id,
            "name": artist_name,
            "role": role,
        })

    return simplified


def simplify_song_detail(song_detail: Dict[str, Any], rank: int) -> Optional[Dict[str, Any]]:
    """提取并整理需要保存的字段。"""

    result: Dict[str, Any] = {}

    result["id"] = song_detail.get("id")
    result["defaultName"] = song_detail.get("defaultName") or song_detail.get("name")
    publishDate = song_detail.get("publishDate")
    result["year"] = extract_year(publishDate)
    result["month"] = extract_month(publishDate)
    result["ratingScore"] = song_detail.get("ratingScore")
    result["favoritedTimes"] = song_detail.get("favoritedTimes")
    result["lengthSeconds"] = song_detail.get("lengthSeconds")
    result["rank"] = rank

    main_pic = song_detail.get("mainPicture") or song_detail.get("MainPicture") or {}
    if isinstance(main_pic, dict):
        result["mainPictureUrlOriginal"] = main_pic.get("urlOriginal")
    else:
        result["mainPictureUrlOriginal"] = None

    artists_raw = song_detail.get("artists") or song_detail.get("Artists") or []
    if not isinstance(artists_raw, list):
        artists_raw = []
    result["artists"] = simplify_artists(artists_raw)

    result["producerNames"] = [
        a["name"] for a in result["artists"]
        if isinstance(a.get("role"), str) and "Producer" in a["role"]
    ]

    tags_raw = song_detail.get("tags") or song_detail.get("Tags") or []
    tags_cleaned = []
    for t in tags_raw:
        if not isinstance(t, dict):
            continue
        tag_obj = t.get("tag") or {}
        if not isinstance(tag_obj, dict):
            tag_obj = {}
        tags_cleaned.append({
            "tagId": tag_obj.get("id"),
            "tagName": tag_obj.get("name"),
            "count": t.get("count"),
        })
    result["tags"] = tags_cleaned

    lyr = choose_original_lyrics(song_detail)
    if lyr:
        codes = lyr.get("cultureCodes") or []
        primary = None
        if isinstance(codes, list) and codes:
            primary = codes[0]
        elif isinstance(codes, str):
            primary = codes

        result["primaryCultureCode"] = primary

        text = lyr.get("value") or ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        result["originalLyrics"] = text.strip()
    else:
        result = None

    return result


def crawl_top_songs(
    top_n: int,
    output_dir: Path,
    page_size: int,
    delay: float,
    max_retries: int,
    rest_every: int,
    rest_seconds: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_ids, max_rank_done = load_existing_state(output_dir)

    session = create_session()

    if top_n <= 0:
        top_n = 489873  # VocaDB 当前最大歌曲数

    if max_rank_done >= top_n:
        logging.info(
            "已完成最大 rank=%d >= 本次目标 top=%d，无需继续。",
            max_rank_done, top_n
        )
        return

    fetched_count = 0
    seen_ranked_ids = max_rank_done
    start = max_rank_done

    while seen_ranked_ids < top_n:
        remaining = top_n - seen_ranked_ids
        page_size_effective = min(page_size, remaining)

        params = {
            "songTypes": "Original",
            "sort": "None",
            "maxResults": page_size_effective,
            "start": start,
            "fields": DETAIL_FIELDS,
            "getTotalCount": "false",
        }

        logging.info(
            "请求歌曲列表：start=%d, maxResults=%d (已 seen=%d / top=%d)",
            start, page_size_effective, seen_ranked_ids, top_n
        )

        data = safe_get_json(session, SONGS_ENDPOINT, params, delay, max_retries)
        if not data:
            logging.error("获取歌曲列表失败，终止。")
            break

        items = data.get("items") or []
        if not items:
            logging.info("列表为空，可能已无更多歌曲，终止。")
            break

        for idx, item in enumerate(items):
            global_rank = start + idx + 1
            song_id = item.get("id")
            if not song_id:
                continue

            seen_ranked_ids += 1
            if seen_ranked_ids > top_n:
                break

            out_path = output_dir / f"song_{song_id}.json"

            if song_id in existing_ids and out_path.exists():
                logging.info("[跳过] rank=%d id=%d 已存在", global_rank, song_id)
                continue

            logging.info("[处理] rank=%d id=%d name=%s",
                         global_rank, song_id,
                         item.get("defaultName") or item.get("name"))

            # 这里直接使用列表返回的 item 作为“详情”
            detail = item
            simplified = simplify_song_detail(detail, rank=global_rank)

            if simplified is None:
                logging.info("[跳过] rank=%d id=%d name=%s （无歌词）",
                             global_rank, song_id,
                             item.get("defaultName") or item.get("name"))
                continue

            with out_path.open("w", encoding="utf-8") as f:
                json.dump(simplified, f, ensure_ascii=False, indent=2)

            fetched_count += 1

            if rest_every > 0 and fetched_count % rest_every == 0:
                logging.info(
                    "已成功下载 %d 首歌，额外休眠 %.1f 秒以减轻服务器压力。",
                    fetched_count, rest_seconds
                )
                time.sleep(rest_seconds)

        start += len(items)

    logging.info(
        "任务结束：目标 top=%d，实际遍历=%d，新下载=%d。",
        top_n, seen_ranked_ids, fetched_count
    )


def main() -> None:
    setup_logger()
    args = parse_args()
    crawl_top_songs(
        top_n=args.top,
        output_dir=Path(args.output_dir),
        page_size=args.page_size,
        delay=args.delay,
        max_retries=args.max_retries,
        rest_every=args.rest_every,
        rest_seconds=args.rest_seconds,
    )


if __name__ == "__main__":
    main()
