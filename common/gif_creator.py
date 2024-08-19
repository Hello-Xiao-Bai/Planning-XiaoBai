from PIL import Image
import os
import sys
import pathlib

root_dir = pathlib.Path(__file__).parent.parent
sys.path.append(str(root_dir))

from common.plot_util import *


def get_gif_path(file_path, gif_name):
    return str(file_path.parent) + "/gif/" + gif_name + ".gif"


def get_image_folder_path(file_path):
    image_folder_path = str(file_path.parent) + "/gif/images"
    if not os.path.exists(image_folder_path):
        os.makedirs(image_folder_path)
    return str(file_path.parent) + "/gif/images"


# 获取图片文件列表
def get_image_files(folder_path):
    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".png"))
    ]
    image_files.sort()
    return image_files


def create_gif(folder_path, output_path, duration=1):
    image_files = get_image_files(folder_path)
    if not image_files:
        print("文件夹中没有.jpg或.png图片文件。")
        return

    try:
        images = []
        for f in image_files:
            try:
                img = Image.open(f)
                images.append(img)
            except Exception as e:
                print(f"无法加载图片文件 '{f}'：{e}")

        if not images:
            print("没有有效的图片文件可供处理。")
            return

        images[0].save(
            output_path,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            optimize=True,
            duration=duration,
            loop=0,
        )
        print("GIF生成成功！")
    except Exception as e:
        print(f"生成GIF时出现错误：{e}")


def delete_folder_contents(folder_path):
    if not os.path.exists(folder_path):
        print(f"路径 {folder_path} 不存在")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"删除 {file_path} 时出错: {e}")


class GifCreator:
    def __init__(self, file_path):
        self.gif_frame = 1000
        self.file_path = file_path
        self.file_name = file_path.stem
        self.image_path = get_image_folder_path(file_path) + "/" + str(self.gif_frame)
        self.image_folder_path = get_image_folder_path(file_path)
        self.gif_path = get_gif_path(self.file_path, str(self.file_name))
        self.clear_gif_folder()

    def get_image_path(self):
        self.gif_frame += 1
        return get_image_folder_path(self.file_path) + "/" + str(self.gif_frame)

    def clear_gif_folder(self):
        delete_folder_contents(self.image_folder_path)

    def create_gif(self, pause_time=PAUSE_TIME):
        print(self.image_folder_path)
        print(self.gif_path)
        create_gif(self.image_folder_path, self.gif_path, duration=pause_time * 1000)
