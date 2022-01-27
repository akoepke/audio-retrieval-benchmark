import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
import os
import shutil


def wait_until_loaded_page(driver, logging):
    """
    Do not start scraping until page loaded fully. Could also use the
    scroll function but we already know the link to fully loaded page.
    """
    ok = 0
    start = time.time()
    while ok == 0:
        try:
            _ = driver.find_element_by_xpath("//*[contains(text(), 'No results found')]")
            ok = 1
        except NoSuchElementException:
            try:
                _ = driver.find_element_by_xpath("//*[contains(text(), 'No more results')]")
                ok = 1
            except NoSuchElementException:
                try:
                    _ = driver.find_element_by_xpath("//*[contains(text(), 'Result limit reached')]")
                    ok = 1
                except NoSuchElementException:
                    time.sleep(1)
    end = time.time()
    logging.info(f"Took {end - start} time to load page")


def wait_until_loaded_page_paid(driver, logging):
    """
    Do not start scraping until page loaded fully. Could also use the
    scroll function but we already know the link to fully loaded page.
    """
    ok = 0
    start = time.time()
    while ok == 0:
        try:
            _ = driver.find_element_by_xpath("//*[contains(text(), 'All Sound Effects')]")
            ok = 1
        except NoSuchElementException:
            time.sleep(1)
    end = time.time()
    logging.info(f"Took {end - start} time to load page")


def wait_for_downloads(zip_videos, logging):
    """
    obtained from
    https://stackoverflow.com/questions/48263317/selenium-python-waiting-for-a-download-process-to-complete-using-chrome-web/48267887
    Script waits for all files to finish downloading before moving to next task
    :param zip_videos: Path to where videos in zip form are downloaded
    """
    logging.info("Waiting for downloads")
    counter = 0
    downloadable = True
    while any([filename.endswith(".crdownload") for filename in
               os.listdir(zip_videos)]):
        for filename in os.listdir(zip_videos):
            if filename.endswith(".crdownload"):
                logging.info(filename)
        time.sleep(2)
        logging.info(".")
        counter += 1
        if counter == 20:
            os.remove(zip_videos / filename)
            logging.info(f"Could not download {filename}")
            downloadable = False
            break

    logging.info("done!")
    return downloadable


def scroll(driver, timeout):
    """
    Scroll until the page cannot be loaded anymore
    """
    time.sleep(4)
    scroll_pause_time = timeout

    # Get scroll height
    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        # Scroll down to bottom
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        driver.find_element_by_xpath('//button[text()="Load more"]').click()

        # Wait to load page
        time.sleep(scroll_pause_time)

        # Calculate new scroll height and compare with last scroll height
        new_height = driver.execute_script("return document.body.scrollHeight")

        if new_height == last_height:
            break
        last_height = new_height


def get_links(main_link, chrome_exe):
    driver = webdriver.Chrome(executable_path=chrome_exe)
    driver.get(main_link)
    timeout = 1
    scroll(driver, timeout)
    soup = BeautifulSoup(driver.page_source)

def move_zip_file(element, zip_videos, file_id, logging):
    shutil.move(zip_videos / element, zip_videos / file_id.lower() / element)
    start = time.time()
    while any([filename.endswith(".zip") for filename in
                os.listdir(zip_videos)]):
        time.sleep(1)
        logging.info(f"Waiting for .zip file to be moved {time.time() - start}")
    logging.info(f"Finished moving {element} to folder {file_id.lower()}")