"""
Synthetic data generator for ForexGuard.
Creates ~50,000 forex platform events with 200 seeded anomalous users.

Anomaly breakdown:
  40 money laundering (deposit -> no trade -> withdraw)
  40 volume spike users
  30 multi-country login users
  30 structuring (small deposits under $10k)
  30 multi-account (shared IP)
  30 night login users
"""

import os
import random
import numpy as np
import pandas as pd
from faker import Faker
from datetime import datetime, timedelta

# reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)

TOTAL_EVENTS = 50_000
NUM_NORMAL_USERS = 800
EVENT_TYPES = [
    "login", "logout", "deposit", "withdrawal",
    "trade_open", "trade_close", "kyc_update",
    "support_ticket", "account_modify"
]
INSTRUMENTS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF", "NZD/USD", "EUR/GBP"]
COUNTRIES = ["US", "UK", "DE", "JP", "AU", "CA", "FR", "SG", "IN", "BR", "NG", "RU", "CN", "AE", "ZA"]

END_DATE = datetime.now()
START_DATE = END_DATE - timedelta(days=90)


def random_timestamp():
    delta = END_DATE - START_DATE
    return START_DATE + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def generate_ip():
    return fake.ipv4()


def generate_device_id():
    return f"DEV-{fake.uuid4()[:8].upper()}"


# user pools (200 anomaly + 800 normal)
all_anomaly_ids = [f"ANOM-{str(i).zfill(4)}" for i in range(200)]
normal_user_ids = [f"USER-{str(i).zfill(4)}" for i in range(NUM_NORMAL_USERS)]

money_laundering_users = all_anomaly_ids[0:40]
volume_spike_users     = all_anomaly_ids[40:80]
multi_country_users    = all_anomaly_ids[80:110]
structuring_users      = all_anomaly_ids[110:140]
multi_account_users    = all_anomaly_ids[140:170]
night_login_users      = all_anomaly_ids[170:200]


def generate_normal_event(user_id):
    """Single normal event with realistic distributions."""
    event_type = random.choice(EVENT_TYPES)
    timestamp = random_timestamp()
    country = random.choice(COUNTRIES[:5])  # normal users stick to few countries
    login_hour = int(np.random.normal(12, 4)) % 24  # mostly daytime

    amount = 0.0
    if event_type == "deposit":
        amount = round(np.random.lognormal(7, 1), 2)
    elif event_type == "withdrawal":
        amount = round(np.random.lognormal(6.5, 0.8), 2)

    trade_volume = round(np.random.lognormal(8, 1.5), 2) if event_type in ["trade_open", "trade_close"] else 0.0
    lot_size = round(random.uniform(0.01, 2.0), 2) if event_type in ["trade_open", "trade_close"] else 0.0
    instrument = random.choice(INSTRUMENTS) if event_type in ["trade_open", "trade_close"] else ""
    session_duration = max(30, int(np.random.normal(600, 200)))
    pnl = round(np.random.normal(0, 500), 2) if event_type == "trade_close" else 0.0

    return {
        "user_id": user_id, "timestamp": timestamp, "event_type": event_type,
        "ip_address": generate_ip(), "device_id": generate_device_id(),
        "country": country, "amount": amount, "trade_volume": trade_volume,
        "trade_instrument": instrument, "lot_size": lot_size,
        "session_duration": session_duration, "login_hour": login_hour, "pnl": pnl
    }


# --- anomaly generators ---

def generate_money_laundering_events(user_id):
    """deposit large -> maybe one tiny trade -> quick withdrawal"""
    events = []
    base_time = random_timestamp()
    ip = generate_ip()
    device = generate_device_id()
    country = random.choice(COUNTRIES[:3])

    events.append({
        "user_id": user_id, "timestamp": base_time, "event_type": "deposit",
        "ip_address": ip, "device_id": device, "country": country,
        "amount": round(random.uniform(15000, 50000), 2),
        "trade_volume": 0.0, "trade_instrument": "", "lot_size": 0.0,
        "session_duration": random.randint(60, 180), "login_hour": base_time.hour, "pnl": 0.0
    })

    if random.random() > 0.3:
        events.append({
            "user_id": user_id, "timestamp": base_time + timedelta(hours=1),
            "event_type": "trade_open", "ip_address": ip, "device_id": device,
            "country": country, "amount": 0.0,
            "trade_volume": round(random.uniform(10, 100), 2),
            "trade_instrument": random.choice(INSTRUMENTS), "lot_size": 0.01,
            "session_duration": random.randint(30, 90), "login_hour": base_time.hour, "pnl": 0.0
        })

    events.append({
        "user_id": user_id, "timestamp": base_time + timedelta(hours=random.randint(2, 12)),
        "event_type": "withdrawal", "ip_address": ip, "device_id": device,
        "country": country, "amount": round(random.uniform(14000, 49000), 2),
        "trade_volume": 0.0, "trade_instrument": "", "lot_size": 0.0,
        "session_duration": random.randint(30, 60), "login_hour": base_time.hour, "pnl": 0.0
    })
    return events


def generate_volume_spike_events(user_id):
    """normal volume for a week, then a sudden 10x spike"""
    events = []
    ip = generate_ip()
    device = generate_device_id()
    country = random.choice(COUNTRIES[:5])
    normal_volume = random.uniform(500, 3000)

    for day in range(7):
        ts = random_timestamp() - timedelta(days=day)
        events.append({
            "user_id": user_id, "timestamp": ts, "event_type": "trade_open",
            "ip_address": ip, "device_id": device, "country": country, "amount": 0.0,
            "trade_volume": round(normal_volume * random.uniform(0.8, 1.2), 2),
            "trade_instrument": random.choice(INSTRUMENTS),
            "lot_size": round(random.uniform(0.1, 1.0), 2),
            "session_duration": random.randint(300, 900),
            "login_hour": random.randint(8, 18), "pnl": round(np.random.normal(0, 200), 2)
        })

    # the spike
    events.append({
        "user_id": user_id, "timestamp": random_timestamp(),
        "event_type": "trade_open", "ip_address": ip, "device_id": device,
        "country": country, "amount": 0.0,
        "trade_volume": round(normal_volume * random.uniform(10, 20), 2),
        "trade_instrument": random.choice(INSTRUMENTS),
        "lot_size": round(random.uniform(5.0, 20.0), 2),
        "session_duration": random.randint(60, 180),
        "login_hour": random.randint(8, 18), "pnl": 0.0
    })
    return events


def generate_multi_country_events(user_id):
    """logins from 6-8 countries within a single day -- physically impossible"""
    events = []
    base_time = random_timestamp()
    device = generate_device_id()
    countries = random.sample(COUNTRIES, k=random.randint(6, 8))

    for country in countries:
        events.append({
            "user_id": user_id,
            "timestamp": base_time + timedelta(hours=random.randint(0, 23)),
            "event_type": "login", "ip_address": generate_ip(),
            "device_id": device, "country": country, "amount": 0.0,
            "trade_volume": 0.0, "trade_instrument": "", "lot_size": 0.0,
            "session_duration": random.randint(60, 300),
            "login_hour": random.randint(0, 23), "pnl": 0.0
        })
    return events


def generate_structuring_events(user_id):
    """many deposits between $500-$9,999 (just under the $10k reporting threshold)"""
    events = []
    ip = generate_ip()
    device = generate_device_id()
    country = random.choice(COUNTRIES[:3])

    for _ in range(random.randint(8, 15)):
        events.append({
            "user_id": user_id, "timestamp": random_timestamp(),
            "event_type": "deposit", "ip_address": ip, "device_id": device,
            "country": country, "amount": round(random.uniform(500, 9999), 2),
            "trade_volume": 0.0, "trade_instrument": "", "lot_size": 0.0,
            "session_duration": random.randint(30, 120),
            "login_hour": random.randint(8, 22), "pnl": 0.0
        })
    return events


def generate_multi_account_events(user_id):
    """multiple accounts on the same IP -- shared IP is set in main()"""
    events = []
    device = generate_device_id()
    country = random.choice(COUNTRIES[:5])

    for _ in range(random.randint(5, 10)):
        events.append({
            "user_id": user_id, "timestamp": random_timestamp(),
            "event_type": random.choice(EVENT_TYPES),
            "ip_address": "SHARED_IP_PLACEHOLDER",
            "device_id": device, "country": country,
            "amount": round(random.uniform(100, 5000), 2) if random.random() > 0.5 else 0.0,
            "trade_volume": round(random.uniform(100, 5000), 2) if random.random() > 0.5 else 0.0,
            "trade_instrument": random.choice(INSTRUMENTS) if random.random() > 0.5 else "",
            "lot_size": round(random.uniform(0.01, 2.0), 2) if random.random() > 0.5 else 0.0,
            "session_duration": random.randint(60, 600),
            "login_hour": random.randint(8, 22), "pnl": 0.0
        })
    return events


def generate_night_login_events(user_id):
    """logins consistently between 1-4 AM"""
    events = []
    ip = generate_ip()
    device = generate_device_id()
    country = random.choice(COUNTRIES[:5])

    for _ in range(random.randint(8, 15)):
        login_hour = random.choice([1, 2, 3, 4])
        ts = random_timestamp().replace(hour=login_hour, minute=random.randint(0, 59))
        events.append({
            "user_id": user_id, "timestamp": ts,
            "event_type": "login", "ip_address": ip, "device_id": device,
            "country": country, "amount": 0.0,
            "trade_volume": round(random.uniform(500, 5000), 2) if random.random() > 0.5 else 0.0,
            "trade_instrument": random.choice(INSTRUMENTS) if random.random() > 0.5 else "",
            "lot_size": round(random.uniform(0.01, 1.0), 2) if random.random() > 0.5 else 0.0,
            "session_duration": random.randint(60, 300),
            "login_hour": login_hour, "pnl": 0.0
        })
    return events


def main():
    print("Generating synthetic forex dataset...")
    all_events = []

    # anomaly events first
    for uid in money_laundering_users:
        all_events.extend(generate_money_laundering_events(uid))
    for uid in volume_spike_users:
        all_events.extend(generate_volume_spike_events(uid))
    for uid in multi_country_users:
        all_events.extend(generate_multi_country_events(uid))
    for uid in structuring_users:
        all_events.extend(generate_structuring_events(uid))

    # multi-account users share IPs in groups of 5
    shared_ips = [generate_ip() for _ in range(6)]
    for i, uid in enumerate(multi_account_users):
        events = generate_multi_account_events(uid)
        shared_ip = shared_ips[i // 5]
        for event in events:
            event["ip_address"] = shared_ip
        all_events.extend(events)

    for uid in night_login_users:
        all_events.extend(generate_night_login_events(uid))

    anomaly_count = len(all_events)

    # fill the rest with normal events
    remaining = TOTAL_EVENTS - anomaly_count
    for _ in range(remaining):
        all_events.append(generate_normal_event(random.choice(normal_user_ids)))

    df = pd.DataFrame(all_events)
    df = df.sort_values("timestamp").reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    df.to_csv("data/raw_events.csv", index=False)

    print(f"Done. {len(df):,} events for {df['user_id'].nunique()} users.")
    print(f"  Anomaly events: {anomaly_count}")
    print(f"  Normal events:  {remaining}")
    print(f"  Anomaly users:  {len(all_anomaly_ids)}")
    print(f"Saved to data/raw_events.csv")


if __name__ == "__main__":
    main()
