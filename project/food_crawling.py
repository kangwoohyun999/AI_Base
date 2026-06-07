# pip install selenium webdriver-manager requests pillow

import os
import json
import time
import requests

from io import BytesIO
from PIL import Image

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


FOODS = {
    "김치찌개": "Kimchi jjigae",
    "된장찌개": "Doenjang jjigae",
    "라면": "Korean ramen",
    "비빔밥": "Bibimbap",
    "삼겹살": "Samgyeopsal"
}

TRAIN_COUNT = 50
TEST_COUNT = 10

BASE_DIR = "food_dataset"

MIN_WIDTH = 150
MIN_HEIGHT = 150

HEADERS = {
    "User-Agent":
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}


def create_driver():

    options = webdriver.ChromeOptions()

    options.add_argument("--start-maximized")
    options.add_argument("--disable-blink-features=AutomationControlled")

    driver = webdriver.Chrome(
        service=Service(
            ChromeDriverManager().install()
        ),
        options=options
    )

    return driver


def scroll_page(driver):

    for _ in range(20):

        driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);"
        )

        time.sleep(2)


def get_bing_image_urls(
    driver,
    keyword,
    target_count
):

    query = keyword.replace(" ", "+")

    url = (
        f"https://www.bing.com/images/search?q={query}"
    )

    driver.get(url)

    time.sleep(3)

    scroll_page(driver)

    image_urls = set()

    elements = driver.find_elements(
        By.CSS_SELECTOR,
        "a.iusc"
    )

    for element in elements:

        try:

            metadata = element.get_attribute("m")

            if not metadata:
                continue

            data = json.loads(metadata)

            image_url = data.get("murl")

            if image_url:
                image_urls.add(image_url)

            if len(image_urls) >= target_count:
                break

        except:
            continue

    return list(image_urls)


def download_image(
    url,
    save_path
):

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=15
        )

        if response.status_code != 200:
            return False

        img = Image.open(
            BytesIO(response.content)
        )

        width, height = img.size

        if width < MIN_WIDTH:
            return False

        if height < MIN_HEIGHT:
            return False

        img.verify()

        with open(save_path, "wb") as f:
            f.write(response.content)

        return True

    except:
        return False


def save_dataset(
    image_urls,
    train_dir,
    test_dir
):

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_saved = 0
    test_saved = 0

    for url in image_urls:

        try:

            lower = url.lower()

            ext = ".jpg"

            if ".png" in lower:
                ext = ".png"

            elif ".webp" in lower:
                ext = ".webp"

            elif ".jpeg" in lower:
                ext = ".jpg"

            if train_saved < TRAIN_COUNT:

                save_path = os.path.join(
                    train_dir,
                    f"{train_saved + 1}{ext}"
                )

                success = download_image(
                    url,
                    save_path
                )

                if success:

                    train_saved += 1

                    print(
                        f"[TRAIN] {train_saved}/{TRAIN_COUNT}"
                    )

            elif test_saved < TEST_COUNT:

                save_path = os.path.join(
                    test_dir,
                    f"{test_saved + 1}{ext}"
                )

                success = download_image(
                    url,
                    save_path
                )

                if success:

                    test_saved += 1

                    print(
                        f"[TEST] {test_saved}/{TEST_COUNT}"
                    )

            if (
                train_saved >= TRAIN_COUNT
                and
                test_saved >= TEST_COUNT
            ):
                break

        except:
            continue

    return train_saved, test_saved


def main():

    os.makedirs(BASE_DIR, exist_ok=True)

    driver = create_driver()

    try:

        for korean_name, english_name in FOODS.items():

            print("\n" + "=" * 60)
            print(f"{korean_name} 수집 시작")
            print("=" * 60)

            urls = []

            # 한글 검색
            korean_urls = get_bing_image_urls(
                driver,
                korean_name,
                300
            )

            print(
                f"한글 검색 URL 수: {len(korean_urls)}"
            )

            urls.extend(korean_urls)

            # 영어 검색
            english_urls = get_bing_image_urls(
                driver,
                english_name,
                300
            )

            print(
                f"영어 검색 URL 수: {len(english_urls)}"
            )

            for url in english_urls:

                if url not in urls:
                    urls.append(url)

            print(
                f"중복 제거 후 총 URL 수: {len(urls)}"
            )

            train_dir = os.path.join(
                BASE_DIR,
                "train",
                korean_name
            )

            test_dir = os.path.join(
                BASE_DIR,
                "test",
                korean_name
            )

            train_saved, test_saved = save_dataset(
                urls,
                train_dir,
                test_dir
            )

            print(
                f"\n{korean_name}"
            )

            print(
                f"Train 저장 완료: {train_saved}장"
            )

            print(
                f"Test 저장 완료: {test_saved}장"
            )

    finally:

        driver.quit()

    print("\n데이터셋 생성 완료")


if __name__ == "__main__":
    main()