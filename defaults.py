import json
from bs4 import BeautifulSoup
class DefaultMixin:
    def getUrlResponse(self,driver,url) -> dict:
        driver.get(url)
        results=driver.page_source
        soup=BeautifulSoup(results,"html.parser")
        response=json.loads(soup.find("pre").text)
        return response
    
          

def calculate_odds(fraction: str) -> float:
    """Convert fractional odds (e.g., '5/2') to decimal odds (e.g., 3.5)"""
    
    numerator, denominator = fraction.split("/")
    return (int(numerator) / int(denominator)) + 1
