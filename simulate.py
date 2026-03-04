import time, random, requests

API = "http://localhost:5000"
USERS = [f"USR-{str(i).zfill(4)}" for i in range(1000, 1050)]
LEGIT_LOCS = [("US","New York, US"),("GB","London, UK"),("DE","Berlin, DE"),("IN","Mumbai, IN"),("CA","Toronto, CA")]
FRAUD_LOCS = [("NG","Lagos, NG"),("RU","Moscow, RU"),("UA","Kyiv, UA"),("GH","Accra, GH")]

def send(tx):
    try:
        r = requests.post(f"{API}/api/transaction", json=tx, timeout=3).json()
        emoji = "🚨" if r.get("risk_level")=="HIGH" else "⚠️ " if r.get("risk_level")=="MEDIUM" else "✅"
        print(f"  {emoji}  {r.get('tx_id','?'):12}  {tx['user_id']:10}  ${tx['amount']:>8.2f}  {tx['location']:20}  Score: {r.get('risk_score','?'):>3}  [{r.get('risk_level','?')}]")
    except Exception as e:
        print(f"  ❌  Error: {e}")
        exit(1)

print("\n  FraudSense Simulator running — Ctrl+C to stop\n")
while True:
    user = random.choice(USERS)
    if random.random() < 0.2:
        c,l = random.choice(FRAUD_LOCS)
        send({"user_id":user,"amount":round(random.uniform(800,8000),2),"country_code":c,"location":l,"device_id":f"dev-unknown-{random.randint(1000,9999)}","is_vpn":random.random()>0.4,"timestamp":time.time()})
    else:
        c,l = random.choice(LEGIT_LOCS)
        send({"user_id":user,"amount":round(random.uniform(20,500),2),"country_code":c,"location":l,"device_id":f"dev-{user}-primary","is_vpn":False,"timestamp":time.time()})
    time.sleep(0.5)
