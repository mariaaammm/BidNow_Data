# BidNow Analytics

Event simulation, tracking specification, and Metabase analytics setup for the **BidNow** platform.

## Overview
This folder contains the tracking plan (`TRACKING_PLAN.md`), synthetic event generator (`simulator.py`), and assets used to build dashboards in Metabase.

`simulator.py` generates realistic auction event streams (searches, item views, item creation, progressive bids, and auction wins) and writes them into a local SQLite database (`auction_analytics.db`, table `user_events`).

## Contents
* `TRACKING_PLAN.md` — event schema, naming conventions, and core dashboard metrics (v1.0).
* `simulator.py` — synthetic event generator creating `auction_analytics.db`.

## How to Run the Simulator
No external Python libraries are required (uses Python standard library only).

```bash
cd analytics
python simulator.py