from zeroconf import ServiceBrowser, Zeroconf
import threading
import socket
import requests
import pandas as pd

class WLEDListener:
    def __init__(self):
        self.devices = {}
        self.lock = threading.Lock()

    def add_service(self, zeroconf, service_type, name):
        info = zeroconf.get_service_info(service_type, name)
        if info and info.addresses:
            ip = socket.inet_ntoa(info.addresses[0])
            with self.lock:
                self.devices[name] = ip

    def update_service(self, zeroconf, service_type, name):
        # Optional: implement update handling if needed
        pass

def discover_wled_devices(timeout=5):
    #devices.drop(columns=devices.columns, inplace=True)
    df = pd.DataFrame(columns=['name', 'ip', 'numLED'])

    zeroconf = Zeroconf()
    listener = WLEDListener()
    service_type = "_wled._tcp.local."

    browser = ServiceBrowser(zeroconf, service_type, listener)

    # Wait a few seconds for discovery
    threading.Event().wait(timeout)

    zeroconf.close()

    # Now query each discovered IP for its WLED name
    for _, ip in listener.devices.items():
        try:
            resp = requests.get(f"http://{ip}/json/info", timeout=2)
            if resp.status_code == 200:
                name = resp.json().get("name", None)
                numLED = resp.json().get("leds", None).get("count", None)
                if name and numLED:
                    df.loc[len(df)] = [name, ip, numLED]
        except requests.RequestException:
            continue  # Skip devices that didn't respond

    return df
