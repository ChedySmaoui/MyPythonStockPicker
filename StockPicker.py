from finvizfinance.screener.overview import Overview
import pandas as pd
import csv
import os

def get_undervalued_stocks():
    """
    Returns a list of tickers with:
    
    - Positive Operating Margin
    - Debt-to-Equity ratio under 1
    - Low P/B (under 1)
    - Low P/E ratio (under 15)
    - Low PEG ratio (under 1)
    - Positive Insider Transactions
    """
    foverview = Overview()
    filters_dict = {'Debt/Equity':'Under 1', 
                    'PEG':'Low (<1)', 
                    'Operating Margin':'Positive (>0%)', 
                    'P/B':'Low (<1)',
                    'P/E':'Low (<15)',
                    'InsiderTransactions':'Positive (>0%)'}
    foverview.set_filter(filters_dict=filters_dict)
    df_overview = foverview.screener_view()
    if not os.path.exists('out'): #ensures you have an 'out' folder ready
        os.makedirs('out')
    df_overview.to_csv('out/Overview.csv', index=False)
    tickers = df_overview['Ticker'].to_list()
    return tickers

print(get_undervalued_stocks())

from transformers import pipeline
import yfinance as yf
from goose3 import Goose
from requests import get

def get_ticker_news_sentiment(ticker):
    """
    Returns a Pandas dataframe of the given ticker's most recent news article headlines,
    with the overal sentiment of each article.

    Args:
        ticker (string)

    Returns:
        pd.DataFrame: {'Date', 'Article title', Article sentiment'}
    """
    ticker_news = yf.Ticker(ticker)
    news_list = ticker_news.get_news()
    extractor = Goose()
    pipe = pipeline("text-classification", model="ProsusAI/finbert")
    data = []
    for dic in news_list:
        title = dic['title']
        response = get(dic['link'])
        article = extractor.extract(raw_html=response.content)
        text = article.cleaned_text
        date = article.publish_date
        if len(text) > 512:
            data.append({'Date':f'{date}',
                         'Article title':f'{title}',
                         'Article sentiment':'NaN too long'})
        else:
            results = pipe(text)
            #print(results)
            data.append({'Date':f'{date}',
                         'Article title':f'{title}',
                         'Article sentiment':results[0]['label']})
    df = pd.DataFrame(data)
    return df

def generate_csv(ticker):
    get_ticker_news_sentiment(ticker).to_csv(f'out/{ticker}.csv', index=False)

undervalued = get_undervalued_stocks()
for ticker in undervalued:
    generate_csv(ticker)