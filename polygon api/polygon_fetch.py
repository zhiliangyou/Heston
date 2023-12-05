import requests # cannot use asyncio and aiohttp for request limit
import pandas as pd
import json
import time
import logging
import argparse
import configparser
import os

class polygon_option_agg_fetcher():
    def __init__(self, base_url, api_key):
        self.base_url = base_url
        self.api_key = api_key
        self.result = {}

    def fetch(self, ticker, multiplier, timespan, start, end, adjusted, sort, limit, max_retry=3, retry_timespan=3):
        count = 0
        logging.info(f'Fetching data for ticker: {ticker}')
        url = f'{self.base_url}/{ticker}/range/{multiplier}/{timespan}/{start}/{end}?adjusted={adjusted}&sort={sort}&limit={limit}&apiKey={self.api_key}'
        while count <= max_retry:
            resp = requests.get(url)
            if resp.status_code == 200:
                self.result[ticker] = resp.json()
                logging.info(f'Data acquired for {ticker}')
                break 
            elif resp.status_code == 429:
                # too many request (free account can only get 5 requests per minute)
                logging.error(f'Limit reached, retrying in 60 seconds')
                time.sleep(60)
            else:
                logging.error(f'resp status code: {resp.status_code}')
                logging.info(f'Retrying in {retry_timespan} seconds...')
                self.result[ticker] = {}
                time.sleep(retry_timespan)
                count += 1

    def save_json(self):
        if not os.path.exists('json'):
            os.mkdir('json')
        for ticker in self.result:
            with open(f'./json/{ticker}.json', 'w') as f:
                json.dump(self.result[ticker], f)

    def save_result_to_csv(self):
        if not os.path.exists('csv'):
            os.mkdir('csv')
        for ticker in self.result:
            ls = self.result[ticker].get('results', '')
            if ls:
                df = pd.DataFrame(ls)
                df['date'] = pd.to_datetime(df['t'], unit='ms')
                df.rename(columns={'v':'volume', 'vw':'vwap', 'o':'open', 'c':'close', 'h':'high', 'l':'low', 't':'timestamp', 'n':'num_trans'}, inplace=True)
                cols = ['date', 'timestamp', 'open', 'high', 'low', 'close', 'num_trans', 'volume', 'vwap']
                df = df[cols]
                df.to_csv(f'./csv/{ticker}.csv')
            else:
                logging.error(f'Empty result for {ticker}')

def main():
    logging.getLogger().setLevel(logging.INFO)
    # Program Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("ini_file", help='ini file name', type=str)

    args = parser.parse_args()
    
    # Read and set arguments
    config = configparser.ConfigParser()
    config.read(args.ini_file)

    polygon_config = config['polygon']
    base_url = polygon_config.get('base_url', '')
    tickers = polygon_config.get('tickers', '').split(',')
    multiplier = polygon_config.get('multiplier', '')
    timespan = polygon_config.get('timespan', '')
    start = polygon_config.get('start', '')
    end = polygon_config.get('end', '')
    adjusted = polygon_config.get('adjusted', '')
    sort = polygon_config.get('sort', '')
    limit = polygon_config.get('limit', '')
    api_key = polygon_config.get('api_key', '')

    fetcher = polygon_option_agg_fetcher(base_url, api_key)

    for ticker in tickers:
        fetcher.fetch(
            ticker, 
            multiplier, 
            timespan, 
            start, 
            end, 
            adjusted, 
            sort, 
            limit,
        )

    fetcher.save_json()
    fetcher.save_result_to_csv()

if __name__ == '__main__':
    main()