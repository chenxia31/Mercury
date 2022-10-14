import os
import re
import requests
import time
def getIPv6Address():
    output = os.popen("ifconfig").read()
    # print(output)
    result = re.findall(r"(([a-f0-9]{1,4}:){7}[a-f0-9]{1,4})", output, re.I)
    return result[0][0]


if __name__ == "__main__":
    # 每60秒获取一次ipv6地州
    temp=0
    while True:
        if temp!=getIPv6Address():
            # WARNING ！！ 这是chenxia的flomo的api，请勿移动
            url='https://flomoapp.com/iwh/NDkwNTUy/00b72c6b1c791765084181b9bf6adc82/'
            data={'content':getIPv6Address()+'#ipv6'}
            requests.post(url=url,data=data)
            temp=getIPv6Address()
        time.sleep(60)

    