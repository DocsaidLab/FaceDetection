import authmecv as acv


def demo_with_ipcam(pipeline, camera_ip):
    demo = acv.WebDemo(camera_ip, pipelines=[pipeline.demo_one_img])
    demo.run()


def demo_with_imgs(pipeline, img_folder, out_folder='out'):
    img_folder = acv.Path(img_folder).absolute()
    if img_folder.is_file():
        img_files = [img_folder]
    else:
        img_files = acv.get_files(img_folder, ['.png', '.jpg', '.jpeg'], recursive=True)
    out_folder = acv.Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)
    pwd = acv.Path('.').absolute()
    for file in acv.Tqdm(img_files):
        img = acv.imread(file)
        img = pipeline.demo_one_img(img)
        out_fpath = out_folder / file.relative_to(pwd)
        out_fpath.parent.mkdir(parents=True, exist_ok=True)
        acv.imwrite(img, out_fpath)
    print('out_folder:', out_folder)
