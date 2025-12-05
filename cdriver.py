from selenium.webdriver import Chrome,ChromeOptions

def chromeDriver(headless:bool) -> Chrome:
    options=ChromeOptions()
    options.add_argument("--no-sandbox")
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option('useAutomationExtension', False)
    driver=Chrome(options=options)
    return driver

