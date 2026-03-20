#!/bin/bash

export MARKET_DATA_PROVIDERS="${MARKET_DATA_PROVIDERS:-stooq}"


TICKERS="AAPL ABBV AMGN AXP BA BAC BIIB BMY C COP CSCO GOOGL GS IBM INTC JNJ JPM KO LLY LMT MCD MDT META MRK MRNA MS MSFT PFE PG REGN UNH V WFC WMT XOM"



# MSFT GOOGL INTC IBM CSCO JNJ UNH MDT C AXP KO PG MCD V

for TICKER in $TICKERS; do
    echo "========================================="
    echo "Starting $TICKER at $(date)"
    echo "========================================="
    
   # Step 1: Download filings + market prices (stooq is default in market.py)
    python -m src.data.extract --ticker $TICKER --num-filings 500 --skip-market-prices
    if [ ! -f "data/processed/${TICKER}_form4.csv" ]; then
        echo "WARNING: Extract failed for $TICKER, skipping"
        sleep 10
        continue
    fi
    
    # python -m src.data.load --load-csv data/processed/${TICKER}_form4.csv

    # # Step 2: Feature engineering
    python -m src.features.pipeline \
        --input data/processed/${TICKER}_form4.csv \
        --skip-preflight
    if [ ! -f "data/processed/${TICKER}_form4_features.csv" ]; then
        echo "WARNING: Features failed for $TICKER, skipping"
        sleep 10
        continue
    fi

    # # Step 3: Earnings dates BEFORE labeling
    python -m src.data.external_events \
        --tickers $TICKER \
        --skip-enforcement \
        --earnings-source sec \
        --max-earnings-rows 60

    # # Step 4: Labeling AFTER earnings are fetched
    python -m src.labels.pipeline \
        --input data/processed/${TICKER}_form4_features.csv \
        --output data/processed/${TICKER}_form4_features_labeled.csv \
        --horizon-days 30 \
        --focus-ticker "$TICKER" \
        --benchmark SPY

    # python -m src.features.pipeline --input data/processed/${TICKER}_form4.csv
    # python -m src.labels.pipeline --input data/processed/${TICKER}_form4_features.csv 

    # python -m src.data.external_events --tickers $TICKER
    
    echo "Done with $TICKER at $(date)"
    echo "Sleeping 5s..."
    sleep 5
done

echo "========================================="
echo "All tickers complete at $(date)"
echo "========================================="
