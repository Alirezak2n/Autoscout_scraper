import os
import time
import logging
import pandas as pd
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import numpy as np


class AutoScout24Scraper:
    def __init__(self, make, model, version, year_from, year_to, power_from, power_to, powertype, zip_code, zipr):
        self.make = make
        self.model = model
        self.version = version
        self.year_from = year_from
        self.year_to = year_to
        self.power_from = power_from
        self.power_to = power_to
        self.powertype = powertype
        self.zip_code = zip_code
        self.zipr = zipr
        self.base_url = (
            "https://www.autoscout24.be/nl/lst/{}/{}/ve_{}?atype=C&cy=B&damaged_listing=exclude&desc=0&"
            "fregfrom={}&fregto={}&powerfrom={}&powerto={}&powertype={}&sort=standard&"
            "source=homepage_search-mask&ustate=N%2CU&zip={}&zipr={}"
        )
        self.listings = []
        self.options = webdriver.ChromeOptions()
        self.options.add_argument("--headless")
        self.options.add_argument("--disable-gpu")
        self.options.add_argument("--no-sandbox")
        self.options.add_argument("--window-size=1920,1080")
        self.options.add_argument("--ignore-certificate-errors")
        self.browser = webdriver.Chrome(options=self.options)
        self.wait = WebDriverWait(self.browser, 10)
        logging.basicConfig(level=logging.INFO)

    def generate_urls(self, num_pages):
        base = self.base_url.format(
            self.make, self.model, self.version, self.year_from, self.year_to,
            self.power_from, self.power_to, self.powertype, self.zip_code, self.zipr
        )
        return [f"{base}&page={i}" if i > 1 else base for i in range(1, num_pages + 1)]

    def scrape(self, num_pages=10, verbose=False):
        for url in self.generate_urls(num_pages):
            try:
                self.browser.get(url)
                self.wait.until(EC.presence_of_element_located(
                    (By.XPATH, "//article[contains(@class, 'cldt-summary-full-item')]")))
                listings = self.browser.find_elements(By.XPATH, "//article[contains(@class, 'cldt-summary-full-item')]")
                for listing in listings:
                    # Title: concatenate all <span> elements inside <h2>
                    try:
                        title_spans = listing.find_elements(By.XPATH, ".//h2/span")
                        title_parts = [span.text.strip() for span in title_spans if span.text.strip()]
                        title = " ".join(title_parts)
                    except Exception:
                        title = None

                    # Seller info
                    try:
                        seller_name = listing.find_element(By.XPATH, ".//span[@data-testid='sellerinfo-company-name']")
                        seller_name = " ".join(seller_name.text.split("\n")).strip()
                        seller_location = listing.find_element(By.XPATH, ".//span[@data-testid='sellerinfo-address']")
                        seller_location = " ".join(seller_location.text.split("\n")).strip()
                    except Exception:
                        seller_name = "Private Seller"
                        seller_location = "Not specified"

                    # Listing URL
                    try:
                        listing_url = listing.find_element(By.XPATH, ".//a").get_attribute("href")
                    except Exception:
                        listing_url = None

                    self.listings.append({
                        "make": listing.get_attribute("data-make"),
                        "model": listing.get_attribute("data-model"),
                        "mileage": listing.get_attribute("data-mileage"),
                        "fuel-type": listing.get_attribute("data-fuel-type"),
                        "first-registration": listing.get_attribute("data-first-registration"),
                        "price": listing.get_attribute("data-price"),
                        "price_label": listing.get_attribute("data-price-label"),
                        "position": listing.get_attribute("data-position"),
                        "seller_name": seller_name,
                        "seller_location": seller_location,
                        "listing_zip_code": listing.get_attribute("data-listing-zip-code"),
                        "listing_url": listing_url,
                        "title": title
                    })

                    if verbose:
                        print(self.listings[-1])

            except TimeoutException:
                logging.warning(f"Timeout while loading {url}")
            except WebDriverException as e:
                logging.error(f"WebDriver exception: {e}")
            time.sleep(0.5)

    def save_to_csv(self, filename="listings.csv"):

        df = pd.DataFrame(self.listings)
        df = enrich_with_drive_and_range(df)
        df.to_csv(filename, index=False)
        logging.info(f"Saved {len(df)} listings to {filename}")

    def quit_browser(self):
        self.browser.quit()

def enrich_with_drive_and_range(dataframe):
    def extract_drive_type(title):
        title = title.lower()
        if "awd" in title:
            return "AWD"
        elif "rwd" in title:
            return "RWD"
        else:
            return "unknown"

    def extract_extended_range(title):
        title = title.lower()
        if "extended" in title or "99" in title or "x" in title:
            return "yes"
        elif "76" in title or "standard range" in title:
            return "no"
        else:
            return "unknown"

    def extract_car_type(title):
        title = title.lower()
        if "gt" in title:
            return "GT"
        elif "awd" in title or "premium" in title:
            return "Premium"
        elif "basic" in title or "base" in title:
            return "Base"
        else:
            return "unknown"

    dataframe["drive_type"] = dataframe["title"].fillna("").apply(extract_drive_type)
    dataframe["extended_range"] = dataframe["title"].fillna("").apply(extract_extended_range)
    dataframe["car_type"] = dataframe["title"].fillna("").apply(extract_car_type)
    return dataframe

def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df.drop_duplicates(inplace=True)
    df = df.dropna(subset=["mileage", "price"])
    df["mileage"] = pd.to_numeric(df["mileage"], errors='coerce')
    df["price"] = pd.to_numeric(df["price"], errors='coerce')
    df = df.dropna(subset=["mileage", "price"])
    df = df[df["mileage"] < 300_000]
    df = df[df["price"] < 150_000]
    df["mileage_grouped"] = (df["mileage"] // 10000) * 10000
    return df


def mileage_price_regression_plot(df):
    grouped = df.groupby("mileage_grouped")["price"].agg(['mean', 'std']).reset_index()
    X = grouped["mileage_grouped"].values.reshape(-1, 1)
    y = grouped["mean"].values

    best_degree = 1
    best_score = float("inf")
    best_model = None
    for degree in range(1, 5):
        poly = PolynomialFeatures(degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        score = mean_squared_error(y, model.predict(X_poly))
        if score < best_score:
            best_score = score
            best_degree = degree
            best_model = (poly, model)

    poly, model = best_model
    X_range = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
    y_pred = model.predict(poly.transform(X_range))

    plt.figure()
    plt.errorbar(X.flatten(), y, yerr=grouped["std"], fmt='o', label='Mean price ± std')
    plt.plot(X_range, y_pred, label=f"Polynomial regression (degree={best_degree})")
    plt.xlabel("Mileage (km)")
    plt.ylabel("Price (€)")
    plt.title("Mileage vs. Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("mileage_price_regression.png")
    plt.show()


def main():
    make = "ford"
    model = "mustang-mach-e"
    version = ""
    year_from = "2021"
    year_to = ""
    power_from = ""
    power_to = ""
    powertype = "kw"
    zip_code = "9000"
    zipr = 1000
    num_pages = 10

    listings_dir = "listings"
    if not os.path.exists(listings_dir):
        os.makedirs(listings_dir)

    csv_file = os.path.join(listings_dir, f"listings_{make}_{model}.csv")

    scraper = AutoScout24Scraper(make, model, version, year_from, year_to, power_from, power_to, powertype, zip_code, zipr)
    scraper.scrape(num_pages=num_pages, verbose=True)
    scraper.save_to_csv(csv_file)
    scraper.quit_browser()
    df = preprocess_data(csv_file)

    mileage_price_regression_plot(df)
    df = df[df["price"]<44000]
    mileage_price_regression_plot(df)


if __name__ == "__main__":
    main()
