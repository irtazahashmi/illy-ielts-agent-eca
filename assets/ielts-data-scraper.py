#!/usr/bin/env python

import level_logging as logging

import argparse
import argcomplete
import csv
import re

from io import TextIOWrapper
from alive_progress import alive_bar

from selenium.webdriver.remote.webelement import WebElement
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By

logger = logging.CustomFormatter.init_logger(__name__)


def check_title(title: str, url: str, sim: float = 0.5) -> bool:
    sim_count = 0

    for word in title.split(" "):
        if word.lower() in url:
            sim_count += 1

    return len(title) > 2 and sim_count / len(title.split(" ")) > sim


def parse_paragraphs(driver: WebDriver, url: str) -> tuple[str, list[str]]:
    title_paths = [
        "/html/body/div[1]/div/div/div[2]/div/article/h3",
        "/html/body/div[1]/div/div/div[2]/div/article/h2[2]",
    ]
    paragraphs = []
    title = ""

    for path in title_paths:
        try:
            title = driver.find_element(By.XPATH, path).text
        except NoSuchElementException:
            continue
        if check_title(title, url):
            break

    paragraphs = driver.find_elements(
        By.XPATH,
        "/html/body/div[1]/div/div/div[2]/div/article/child::p",
    )

    slices: dict[int, int] = {}
    matches: list[str] = []
    current_start = -1

    # Filter out all the wrong paragraphs
    for index, paragraph in enumerate(paragraphs):
        begin_answer = re.search("\\w+ Answer.*:", paragraph.text)

        end_answer = re.search("Discussion.*:", paragraph.text)
        end_answer = re.search("[T,t]alk about the following topics", paragraph.text)

        if begin_answer is not None:
            matches.append(begin_answer[0])

            if current_start != -1:
                slices[current_start] = index

            current_start = index
        elif current_start == -1:
            continue
        elif len(paragraph.text.strip()) == 0:
            slices[current_start] = index
            current_start = -1
        elif end_answer is not None:
            slices[current_start] = index
            current_start = -1

        if (
            current_start != -1
            and index == len(paragraphs) - 1
            and current_start not in slices.keys()
        ):
            slices[current_start] = index + 1
            current_start = -1

    parsed_paragraphs: list[str] = []
    # logger.debug("Slices: {}\nMatches: {}".format(slices, matches))

    for index, key in enumerate(slices.keys()):
        matched_paragraphs = [
            paragraph.text for paragraph in paragraphs[key : slices[key]]
        ]
        matched_paragraphs[0] = matched_paragraphs[0].removeprefix(matches[index])

        for paragraph in matched_paragraphs:
            # In case there is only one paragraph split by <br>
            for split_paragraph in paragraph.split("\n"):
                if len(split_paragraph) != 0:
                    number_prefix_match = re.search(split_paragraph, "\\d+\\.")

                    if number_prefix_match is not None:
                        split_paragraph = split_paragraph.removeprefix(
                            number_prefix_match[0]
                        )

                    parsed_paragraphs.append(split_paragraph)

    return title, parsed_paragraphs


def get_next(driver: WebDriver) -> WebElement | None:
    next = None

    try:
        next = driver.find_element(By.XPATH, "//li[contains(@class, 'next')]/a")
    except NoSuchElementException:
        pass
    finally:
        return next


def append_sample_to_corpus(
    sample: tuple[str, list[str]], output_file: str | TextIOWrapper
):
    title, samples = sample
    compressed_samples = "|".join(samples)
    formatted = [title, compressed_samples]

    if type(output_file) is TextIOWrapper:
        w = csv.writer(output_file)
        w.writerow(formatted)
    elif type(output_file) is str:
        with open(output_file, "a") as f:
            w = csv.writer(f)
            w.writerow(formatted)
    else:
        raise ValueError("output_file must be either a str or TextIOWrapper")


def write_corpus(corpus: dict[str, list[str]], output_file: str):
    with open(output_file, "w") as f:
        for title in corpus.keys():
            w = csv.writer(f)
            append_sample_to_corpus((title, corpus[title]), f)


def main(
    url: str,
    output_file: str,
    number_of_samples: int,
    run_headless: bool,
    incremental: bool,
    verbose: bool,
):
    logger.info("Starting...")

    chrome_options = Options()

    if run_headless:
        logger.info("Running in headless mode")
        chrome_options.add_argument("--headless")

    driver = webdriver.Chrome("chromedriver", options=chrome_options)

    corpus = {}
    successful_samples = 0

    try:
        with alive_bar(number_of_samples, bar="halloween", spinner="frank") as bar:
            for successful_samples in range(number_of_samples):
                driver.get(url)
                bar()

                try:
                    title, paragraphs = parse_paragraphs(driver, url)
                except Exception as e:
                    logger.error("Failed to parse the following url: {}".format(url))

                    if verbose:
                        logger.debug(e, exc_info=True)
                else:
                    if len(paragraphs) == 0:
                        logger.warning(
                            "No sample found for title: '{}', url: {}".format(
                                title, url
                            )
                        )

                    corpus[title] = paragraphs

                    if incremental:
                        append_sample_to_corpus((title, paragraphs), output_file)

                next_element = get_next(driver)

                if next_element is None:
                    break

                url = next_element.get_attribute("href")
    except KeyboardInterrupt:
        logger.warning("Procedure Interrupted manually.")

    finally:
        if successful_samples != number_of_samples - 1:
            logger.info(
                "Stopped early, could not complete {} samples".format(
                    number_of_samples - successful_samples - 1
                )
            )
        if not incremental:
            logger.info("Writing to {}...".format(output_file))
            write_corpus(corpus, output_file)

        logger.info("Saved to {}".format(output_file))


if __name__ == "__main__":
    # Construct an argument parser
    all_args = argparse.ArgumentParser()

    # Add arguments to the parser
    all_args.add_argument(
        "-n",
        "--number-of-samples",
        required=False,
        type=int,
        default=1,
        help="Number of samples to be taken from the website (default 1)",
    )
    all_args.add_argument(
        "-o",
        "--output-file",
        required=True,
        help="Name of the data dump (should be .csv)",
    )
    all_args.add_argument(
        "-u",
        "--url",
        required=True,
        help="Target website (cannot be included in repo)",
    )
    all_args.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        help="Run Selenium in headless mode (don't display browser GUI)",
    )
    all_args.add_argument(
        "--incremental",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Incrementally write data to ouput file (default True)",
    )
    all_args.add_argument(
        "--verbose",
        "-v",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include stacktrace in failed url parsing passes (default False)",
    )
    argcomplete.autocomplete(all_args)
    args = vars(all_args.parse_args())

    main(
        args["url"],
        args["output_file"],
        args["number_of_samples"],
        args["headless"],
        args["incremental"],
        args["verbose"],
    )
