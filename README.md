# akari_tracking_palette

俯瞰の位置情報のパレット上に図形を書くことで、監視エリアを指定できるアプリ

## submoduleの更新
`git submodule update --init --recursive`  

## 仮想環境の作成
`python -m venv venv`  
`source venv/bin/activate`  
`pip install -r requirements.txt`  

## サーバの起動
`python3 server.py`  

## 使い方
・counterの起動
`python3 tracking_counter.py`  

俯瞰図のウインドウが起動するので、ここで図形を描くと、図形のエリア内の物体がカウントされる。  

キーボード操作は下記の通り  
- `r`: 四角形モードに設定。左クリックして離すまでの間の2点をエリアにする。  
- `c`: 円モードに設定。左クリックで円の中心を決め、離すと半径が決定。  
- `d`: 描画したエリアを全て削除。  
- `s`: 今のエリアを`rect_json`以下にsaveする。  
- `0`: エリアIDを0にする。  
- `1`: エリアIDを1にする。  
- `2`: エリアIDを2にする。  

## saveしたjsonでの起動
・下記でsaveしたjsonを元にcounterの起動が可能
`python3 tracking_counter.py`  
