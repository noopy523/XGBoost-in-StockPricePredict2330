`XGBoost`  `VSCode` 
# Stock Price Predict in 2330  
* NOTE : 記得資料檔案和.py要放在同一個資料夾

* 股票資料下載 (https://finance.yahoo.com/quote/2330.TW?p=2330.TW&.tsrc=fin-srch)
* 步驟
  * 第一步：獲取數據
  * 第二步：特徵工程
  * 第三步：數據標準化，計算出標準化的數據，在對其進行數據分割
  * 第四步：生成訓練資料和測試資料。因訓練資料和測試資料的標準化方式不同，因此要切分訓練和測試資料。
  * 第五步：標籤和特徵的標準化，目的是為了對在訓練集不能代表全體的情形下，使模型正確運行的一種技巧
  * 第六步：建立樣本
  * 第七步：開始訓練
  * 第八步：結果

    ```
    結果呈現如下
    ``` 
![image](https://github.com/noopy523/XGBoost-in-StockPricePredict2330/blob/main/result.jpg))

