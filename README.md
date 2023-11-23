# akari_tracking_palette

YOLOの3次元物体認識を俯瞰図にプロットし、俯瞰図上に図形を書くことで、好きな監視エリアを指定できるアプリ。
オリジナルのYOLOモデルを使用することができます。
認識モデルの作り方は下記を参照ください。
https://akarigroup.github.io/docs/source/dev/custom_object_detection/main.html


## submoduleの更新
`git submodule update --init --recursive`  

## 仮想環境の作成
`python -m venv venv`  
`source venv/bin/activate`  
`pip install -r requirements.txt`  

## 使い方

アプリ実行時に物体認識のウインドウと俯瞰図のウインドウが起動する。この俯瞰図内にエリアの図形を描くと、各エリア内にいる人数がカウントされる。  
マウスの左クリックを押して、離すまでの間にマウスを動かすことで、任意の四角形、円を描画でき、その図形内をエリアとして設定できる。  

キーボード操作は下記の通り  
- `r`: 四角形モードに設定。左クリックして離すまでの間の2点をエリアにする。  
- `c`: 円モードに設定。左クリックで円の中心を決め、離すと半径が決定。  
- `d`: 描画したエリアを全て削除。  
- `s`: 今のエリアを`rect_json`ディレクトリにsaveする。  
- `0`: エリアIDを0にする。  
- `1`: エリアIDを1にする。  
- `2`: エリアIDを2にする。  

## 引数
- `-m`, `--model`: オリジナルのYOLO認識モデル(.blob)を用いる場合にパスを指定。引数を指定しない場合、YOLO v4のCOCOデータセット学習モデルを用いる。  
- `-c`, `--config`: オリジナルのYOLO認識ラベル(.json)を用いる場合にパスを指定。引数を指定しない場合、YOLO v4のCOCOデータセット学習ラベルを用いる。  
- `-f`, `--fps`: カメラ画像の取得PFS。デフォルトは8。OAK-Dの性質上、推論の処理速度を上回る入力を与えるとアプリが異常終了しやすくなるため注意。  
- `-d`, `--display_camera`: この引数をつけると、RGB,depthの入力画像も表示される。  
- `-roi_path`: 指定したパスのjsonファイルから注目エリアを読みこむ。サンプルとして、`roi_json/sample/sample.json`が最初から用意されている。またアプリ内で書いたエリアを保存して次回以降使用することも可能。  

## サンプルアプリ

### 監視サンプル
`python3 monitor_sample.py`  

各エリア内にある認識した物体の数をターミナルに表示する。

### 挨拶サンプル
`python3 greeting_sample.py`  

エリア0に人がいたら「こんにちは」、エリア0で挨拶した人がエリア1に移動したら「さようなら」とAKARIのM5に表示する。  
`roi_json/sample/sample.json`を使用して起動してみることを推奨。  
`python3 greeting_sample.py --roi_path roi_json/sample/sample.json`  


### 人数カウント
`python3 person_counter.py`  

各エリア内の人数をカウントし、AKARIのM5に表示する。
上記のキーボード操作に加え、下記のキー入力でヘッドを動かすことが可能。

- `l`: ヘッドが右に動く。  
- `j`: ヘッドが左に動く。  
- `i`: ヘッドが上に動く。  
- `m`: ヘッドが下に動く。  
- `k`: ヘッドが初期位置に戻る  
