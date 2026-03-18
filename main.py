from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import shutil

# 导入你刚刚写好的终极版搜索器
from search import load_system, search_music

app = FastAPI(title="音乐音频相似度检索系统 API")

# 配置 CORS 跨域（非常关键！否则后续 Vue 前端会被浏览器拦截报错）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有前端来源访问
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("⏳ 正在初始化全局检索系统...")
# 在服务器启动时，只加载一次模型和 Faiss 索引到内存中
# 这样用户搜索时就不会卡顿了
INDEX, MODEL = load_system()
print("✅ Web 服务器后台准备就绪！")


@app.post("/search")
async def api_search_music(file: UploadFile = File(...)):
    """
    接收前端上传的音频文件，进行听歌识曲
    """
    # 1. 将用户上传的音频临时存到本地
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # 2. 调用你之前写好的核心搜索逻辑 [cite: 378-379]
        # 这里默认返回最相似的 5 首歌
        results = search_music(temp_file_path, INDEX, MODEL, top_k=5)

        # 3. 构造标准的 JSON 格式返回给前端 [cite: 122-123]
        return {
            "code": 200,
            "message": "检索成功",
            "data": results
        }
    except Exception as e:
        return {
            "code": 500,
            "message": f"处理音频时发生错误: {str(e)}",
            "data": []
        }
    finally:
        # 4. 无论成功与否，都要把临时文件删掉，防止服务器硬盘被撑爆
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


if __name__ == "__main__":
    # 启动服务器，运行在本地 8000 端口
    uvicorn.run(app, host="127.0.0.1", port=8000)