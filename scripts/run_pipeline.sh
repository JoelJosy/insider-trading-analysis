#!/bin/bash

export MARKET_DATA_PROVIDERS="${MARKET_DATA_PROVIDERS:-stooq}"


TICKERS="ABBV BAC BIIB COP JPM LLY META MRK MRNA PFE REGN WMT XOM"

for TICKER in $TICKERS; do
    echo "========================================="
    echo "Starting $TICKER at $(date)"
    echo "========================================="
    
    python -m src.data.extract --ticker $TICKER --num-filings 500 --skip-market-prices
    
    if [ ! -f "data/processed/${TICKER}_form4.csv" ]; then
        echo "WARNING: Extract failed for $TICKER, skipping"
        continue
    fi
    
    # python -m src.data.load --load-csv data/processed/${TICKER}_form4.csv
    python -m src.features.pipeline --input data/processed/${TICKER}_form4.csv
    python -m src.labels.pipeline --input data/processed/${TICKER}_form4_features.csv
    
    echo "Done with $TICKER at $(date)"
done

echo "All tickers complete"
python -m src.data.external_events --tickers BIIB REGN MRNA NVDA META BAC WMT