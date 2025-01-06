import requests
import base64
import time
from PIL import Image
from io import BytesIO

# 配置
RUNPOD_API_KEY = "rpa_DV3RBXXWXMDRNCPNYRIZRFVXETYAS8XKQY9WZXTRttxrnu"
ENDPOINT_ID = "890dnquzcauwm5"
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

headers = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json"
}

def prepare_image(image_path):
    """准备base64编码的图像"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()

def run_inference(image_b64):
    """发送推理请求"""
    payload = {
        "input": {
            "image": image_b64,
            "prompt": "a beautiful landscape, high quality, detailed"
        }
    }
    
    response = requests.post(
        f"{BASE_URL}/run",
        headers=headers,
        json=payload
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"推理请求失败: {response.status_code}, {response.text}")

def check_status(task_id):
    """检查任务状态"""
    response = requests.get(
        f"{BASE_URL}/status/{task_id}",
        headers=headers
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"状态检查失败: {response.status_code}, {response.text}")

def main():
    # 准备测试图像
    try:
        image_b64 = prepare_image("ic_bedroom.jpg")
    except FileNotFoundError:
        print("错误: 找不到测试图像 ic_bedroom.jpg")
        return
    
    # 发送推理请求
    print("发送推理请求...")
    try:
        run_response = run_inference(image_b64)
        task_id = run_response.get("id")
        if not task_id:
            print("错误: 未获取到任务ID")
            return
    except Exception as e:
        print(f"推理请求失败: {str(e)}")
        return
    
    # 轮询检查状态
    print("等待结果...")
    max_attempts = 60  # 最多等待60秒
    while max_attempts > 0:
        try:
            status_response = check_status(task_id)
            status = status_response.get("status")
            
            if status == "COMPLETED":
                # 获取结果
                output = status_response.get("output")
                if output and "output" in output:
                    # 保存图像
                    img_data = base64.b64decode(output["output"])
                    img = Image.open(BytesIO(img_data))
                    img.save("output.png")
                    print("成功! 生成的图像已保存为 output.png")
                else:
                    print("错误: 输出结果格式不正确")
                return
            elif status == "FAILED":
                print(f"任务失败: {status_response.get('error')}")
                return
            
        except Exception as e:
            print(f"检查状态时出错: {str(e)}")
            return
            
        time.sleep(1)
        max_attempts -= 1
    
    print("错误: 等待超时")

if __name__ == "__main__":
    main()