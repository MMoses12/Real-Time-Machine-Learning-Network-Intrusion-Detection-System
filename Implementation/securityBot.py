from tensorflow.keras.models import load_model
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scapy.all import sniff, IP, TCP, UDP
import pandas as pd
from threading import Thread
import joblib
import os
import socket
from collections import defaultdict

# === Load Trained Models and Encoders ===
autoencoder = load_model('trained_autoencoder.h5', custom_objects={'mse': 'mean_squared_error'})
scaler = joblib.load('trained_scaler.save')
proto_encoder = joblib.load('proto_encoder.pkl')
state_encoder = joblib.load('state_encoder.pkl')
ensemble_model = joblib.load('ensemble_model.pkl')
classifier_scaler = joblib.load('classifier_scaler.pkl')
attack_encoder = joblib.load('attack_encoder.pkl')
feature_columns = joblib.load('classifier_feature_columns.pkl')

# === Configuration ===
BLOCK_DURATION = 100
INTERFACE = "ens33"
SYN_FLOOD_THRESHOLD = 20
SYN_WINDOW = 1.0

# === Get Local IP Address ===
def get_host_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        return s.getsockname()[0]
    except:
        return '127.0.0.1'
    finally:
        s.close()

HOST_IP = get_host_ip()
print(f"Host IP: {HOST_IP}")

# === Runtime State ===
flows = {}
output = []
blocked_ips = set()
ip_unblock_times = {}
syn_tracker = defaultdict(list)

# === Block/Unblock ===
def block_ip_os(ip):
    print(f"[BLOCK] Blocking {ip}")
    os.system(f"sudo iptables -I INPUT -s {ip} -j DROP")

def unblock_ip_os(ip):
    print(f"[UNBLOCK] Unblocking {ip}")
    os.system(f"sudo iptables -D INPUT -s {ip} -j DROP")

# === SYN Flood Detection ===
def detect_syn_flood(ip):
    now = time.time()
    syn_tracker[ip] = [t for t in syn_tracker[ip] if now - t < SYN_WINDOW]
    return len(syn_tracker[ip]) > SYN_FLOOD_THRESHOLD

# === Packet Handler ===
def get_flow_key(pkt):
    ip = pkt[IP]
    src_ip, dst_ip = ip.src, ip.dst
    src_port = pkt.sport if hasattr(pkt, 'sport') else 0
    dst_port = pkt.dport if hasattr(pkt, 'dport') else 0
    proto = ip.proto
    return (src_ip, dst_ip, src_port, dst_port, proto)

def process_packet(pkt):
    if IP not in pkt or pkt[IP].dst != HOST_IP:
        return

    key = get_flow_key(pkt)
    attacker_ip = key[0]
    if attacker_ip in blocked_ips:
        return

    if TCP in pkt and pkt[TCP].flags == 'S':
        syn_tracker[attacker_ip].append(time.time())
        if detect_syn_flood(attacker_ip):
            print(f"[ALERT] SYN flood or stealth scan from {attacker_ip}")
            blocked_ips.add(attacker_ip)
            ip_unblock_times[attacker_ip] = time.time() + BLOCK_DURATION
            block_ip_os(attacker_ip)
            return

    rev_key = (key[1], key[0], key[3], key[2], key[4])
    ip = pkt[IP]
    size = len(pkt)
    now = time.time()

    proto_map = {6: 'tcp', 17: 'udp'}
    proto = proto_map.get(ip.proto, 'other')

    if key not in flows and rev_key not in flows:
        flows[key] = {
            'start': now, 'last': now,
            'spkts': 1, 'dpkts': 0,
            'sbytes': size, 'dbytes': 0,
            'sttl': ip.ttl, 'dttl': 0,
            'proto': proto,
            'state': 'SYN' if TCP in pkt and pkt[TCP].flags.S else '-'
        }
    else:
        fkey = key if key in flows else rev_key
        flow = flows[fkey]
        flow['last'] = now
        if key in flows:
            flow['spkts'] += 1
            flow['sbytes'] += size
            flow['sttl'] = ip.ttl
            if TCP in pkt and pkt[TCP].flags.F:
                flow['state'] = 'FIN'
        else:
            flow['dpkts'] += 1
            flow['dbytes'] += size
            flow['dttl'] = ip.ttl

# === Sniff Thread ===
def start_sniffing():
    sniff(filter=f"ip and dst host {HOST_IP}", iface=INTERFACE, prn=process_packet, store=0)

Thread(target=start_sniffing, daemon=True).start()
print("Detection system started. Press Ctrl+C to stop.")

# === Main Loop ===
try:
    while True:
        now = time.time()

        # Unblock expired IPs
        for ip in list(ip_unblock_times):
            if now > ip_unblock_times[ip]:
                unblock_ip_os(ip)
                blocked_ips.discard(ip)
                del ip_unblock_times[ip]

        # Flush expired flows
        expired = []
        for fkey, flow in list(flows.items()):
            if now - flow['last'] > 0.2:
                if fkey[0] in blocked_ips:
                    expired.append(fkey)
                    continue

                dur = flow['last'] - flow['start']
                sload = (flow['sbytes'] * 8) / dur if dur else 0
                dload = (flow['dbytes'] * 8) / dur if dur else 0
                rate = (flow['spkts'] + flow['dpkts']) / (dur + 1e-6)
                sinpkt = flow['spkts'] / (dur + 1e-6)
                dinpkt = flow['dpkts'] / (dur + 1e-6)
                spd = flow['spkts'] / (dur + 1e-6)
                dpd = flow['dpkts'] / (dur + 1e-6)

                try:
                    proto = proto_encoder.transform([flow['proto']])[0]
                    state = state_encoder.transform([flow['state']])[0]
                except:
                    continue

                row = [dur, flow['spkts'], flow['dpkts'], flow['sbytes'], flow['dbytes'], rate,
                       flow['sttl'], flow['dttl'], sload, dload, sinpkt, dinpkt, spd, dpd, proto, state]
                output.append((fkey, row))
                expired.append(fkey)

        for fkey in expired:
            del flows[fkey]

        if not output:
            time.sleep(0.2)
            continue

        # === Filter Out Blocked IPs ===
        filtered = [(k, row) for k, row in output if k[0] not in blocked_ips]
        output.clear()

        if not filtered:
            time.sleep(0.2)
            continue

        flow_keys, data_rows = zip(*filtered)
        df = pd.DataFrame(data_rows, columns=feature_columns)

        auto_features = [col for col in df.columns if col not in ['proto', 'state']]
        scaled_auto = scaler.transform(df[auto_features])
        reconstructed = autoencoder.predict(scaled_auto)
        errors = np.mean((scaled_auto - reconstructed) ** 2, axis=1)
        threshold = np.percentile(errors, 90)
        auto_pred = (errors > threshold).astype(int)

        scaled_class = classifier_scaler.transform(df[feature_columns])
        class_pred = ensemble_model.predict(scaled_class)
        attack_types = attack_encoder.inverse_transform(class_pred)

        for i in range(len(df)):
            src, dst = flow_keys[i][0], flow_keys[i][1]
            if auto_pred[i]:
                print(f"[ALERT] {src} -> {dst} | Classifier Attack Type: {attack_types[i]}")
                #  | Classifier Attack Type: {attack_types[i]}
                if src not in blocked_ips:
                    blocked_ips.add(src)
                    ip_unblock_times[src] = time.time() + BLOCK_DURATION
                    block_ip_os(src)
                flows = {k: v for k, v in flows.items() if k[0] != src}
            else:
                print(f"[OK] {src} → {dst} | Normal flow")

        time.sleep(0.2)

except KeyboardInterrupt:
    print("\nDetection stopped.")