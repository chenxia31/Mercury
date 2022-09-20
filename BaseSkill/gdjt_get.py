# 希望利用浏览器抓包来实现报名的整个过程
#chrome- F12-i network-XHR，看到post，以及对应的response中，result.json、push.json\result.json
# 关系push和result
# 利用程序来模拟push.json这个动作，这里可以看到post得地址

# 查看push里面的header
    # url
    # content-type
    # agent-type
    # cookie 隐私信息，是需要的
    # referer
    # token 每次提交对应的token都会发生变化

# 查看push里面的request
    # act_str 加密的信息

# 查看result里面的状态
    # host
    # filename /Swooze/result.json
    # act_id=''

# 探索一：看一下这个token是用来做什么的，在debugger中看token
# token:gt.encryptParam(''.concat(~~(+new/Date/1000).'@tj'))
# act_str=

# 现在就需要看encryptParam的作用
# coding；utf-8
from sqlite3 import connect
import requests
import base64
from Crypto.Cipher import AES
import time
import json

class_id='14984'
url_id='http://gdjt.tongji.edu.cn/Swoole/push.json'
url_result='http://gdjt.tongji.edu.cn/Swoole/result.json/?act_id='+class_id

# 其中关键的id
headers={
    "Referer":'http://gdjt.tongji.edu.cn/PC/',
    'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/105.0.0.0 Safari/537.36',
    'Token':'Token' #后面update_token_str()中会随着时间实时更新
}

cookie_str='PHPSESSID=dmir3vrasi1ntfu2u9t9sfia0h; Hm_lvt_7543ad4d5aca565a9ba0da0cc74c4eb7=1663375191; Hm_cv_7543ad4d5aca565a9ba0da0cc74c4eb7=1*visitor*PC; pwapp_unbs=10212970; pwapp_ext_sid=2233391; pwapp_uname=%E5%BE%90%E6%99%A8%E9%BE%99; tokens=7713a452e063e87b4bdc1e87e0ef4595; connection_id=3ddd1c76e980a2f36f8caa7f51a4df8b; IS_YXMC=10212970; think_language=zh-CN; Hm_lpvt_7543ad4d5aca565a9ba0da0cc74c4eb7=1663379141'

# 将cookie转换为key- value形式
cookies={}
for line in cookie_str.split(';'):
    key,value=line.split('=',1)
    cookies[key]=value

data={}
data['act_str']='act_str'

def update_token_str():
    # 这里根据网页的源代码进行编写
    # AES加密、Key寻找、token生成、act_id生成

    # 这是明文保存在代码中
    key=b"xiaofaai@act_id!"
    iv=b"tongjixf@act_id!"

    unix_time=int(time.time())

    token_str= "\""+str(unix_time)+"@tj"+"\""
    print('token_str',token_str)
    base64_token_string_str=(base64.b64encode(bytes(token_str,encoding='utf-8'))).decode('utf-8')

    pad=lambda s:s+(16-len(s)%16)*chr(0)
    data_AES=pad(base64_token_string_str)
    cipher=AES.new(key,AES.MODE_CBC,iv)
    encrypted_bytes=cipher.encrypt(data_AES.encode('utf-8'))
    token_AES=(base64.b64encode(encrypted_bytes)).decode('utf-8')
    print(token_AES)
    base64_token_aes_str=(base64.b64encode(bytes(token_AES,encoding='utf-8'))).decode('utf-8')
    headers['Token']=base64_token_aes_str
    print(base64_token_aes_str)

    # act_str aes 生成
    act_str_raw='{\'act_id\':\''+class_id+'\',\'buy_ticket_type\':\'PC\',\'time\':'+str(unix_time)+'}'
    base64_act_str_raw_str=(base64.b64encode(bytes(act_str_raw,encoding='utf-8'))).decode('utf-8')
    cipher=AES.new(key,AES.MODE_CBC,iv)
    encrypted_bytes_act=cipher.encrypt((base64_act_str_raw_str).encode('utf-8'))
     # 这里似乎是不需要padding
    act_str_AES=(base64.b64encode(encrypted_bytes_act)).decode('utf-8')
    bytes_act_str_AES=act_str_AES.encode('utf-8')
    base64_act_strr_AES_str=(base64.b64encode(bytes(act_str_AES,encoding='utf-8'))).decode('utf-8')
    data['act_str']=base64_act_strr_AES_str
    print(base64_act_strr_AES_str)


while True:
    try:
        print('Current class ID=',class_id)

        update_token_str()
        
        print(data)
        resp=requests.post(url_id,headers=headers,cookies=cookies,data=data,timeout=600)
        # 600十分钟

        return_string=resp.content.decode('utf-8')

        print('status=',resp.status_code)
        print('return string=',return_string)

        if return_string.find('排队中')!=-1:
            print('waiting...')
            time.sleep(1)

            return_string_result=requests.post(url_result,headers=headers,cookies=cookies,timeout=600)
            return_string_result=return_string_result.content.decode('utf-8')
            print(return_string_result)

            if return_string_result.find('报名成功') !=-1:
                print('Success! with class id=',class_id)
                exit(0)
        else:
            time.sleep(10)
    except requests.exceptions.ConnectionError as e:
        print('error')
        time.sleep(10)
ou