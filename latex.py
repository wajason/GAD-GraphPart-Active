import re
import collections

# ======================================================
# 1. 請將你的實驗 Log 完整貼在下方的 raw_data 引號內
# ======================================================
raw_data = """
>>> Running: gcn | random | weibo | Budget=10
RESULT: (weibo|gcn|random|B=10) AUC: 0.1136 ± 0.0627 | Best: 0.2820 | Found: 0.3/10

>>> Running: gcn | random | weibo | Budget=20
RESULT: (weibo|gcn|random|B=20) AUC: 0.1292 ± 0.0568 | Best: 0.2415 | Found: 0.5/20

>>> Running: gcn | random | weibo | Budget=40
RESULT: (weibo|gcn|random|B=40) AUC: 0.1908 ± 0.1601 | Best: 0.6157 | Found: 0.9/40

>>> Running: gcn | random | weibo | Budget=80
RESULT: (weibo|gcn|random|B=80) AUC: 0.5125 ± 0.1739 | Best: 0.8410 | Found: 3.1/80

>>> Running: gcn | random | weibo | Budget=160
RESULT: (weibo|gcn|random|B=160) AUC: 0.6170 ± 0.1189 | Best: 0.8007 | Found: 6.4/160

>>> Running: gcn | random | weibo | Budget=320
RESULT: (weibo|gcn|random|B=320) AUC: 0.7231 ± 0.0829 | Best: 0.8349 | Found: 13.6/320

>>> Running: gcn | uncertainty | weibo | Budget=10
RESULT: (weibo|gcn|uncertainty|B=10) AUC: 0.2651 ± 0.1158 | Best: 0.5812 | Found: 2.2/10

>>> Running: gcn | uncertainty | weibo | Budget=20
RESULT: (weibo|gcn|uncertainty|B=20) AUC: 0.2145 ± 0.1332 | Best: 0.5785 | Found: 2.0/20

>>> Running: gcn | uncertainty | weibo | Budget=40
RESULT: (weibo|gcn|uncertainty|B=40) AUC: 0.2021 ± 0.1717 | Best: 0.6157 | Found: 1.6/40

>>> Running: gcn | uncertainty | weibo | Budget=80
RESULT: (weibo|gcn|uncertainty|B=80) AUC: 0.2396 ± 0.1718 | Best: 0.6416 | Found: 3.9/80

>>> Running: gcn | uncertainty | weibo | Budget=160
RESULT: (weibo|gcn|uncertainty|B=160) AUC: 0.4117 ± 0.2635 | Best: 0.8231 | Found: 15.7/160

>>> Running: gcn | uncertainty | weibo | Budget=320
RESULT: (weibo|gcn|uncertainty|B=320) AUC: 0.5456 ± 0.1819 | Best: 0.7444 | Found: 47.3/320

>>> Running: gcn | degree | weibo | Budget=10
RESULT: (weibo|gcn|degree|B=10) AUC: 0.1448 ± 0.0888 | Best: 0.2894 | Found: 0.0/10

>>> Running: gcn | degree | weibo | Budget=20
RESULT: (weibo|gcn|degree|B=20) AUC: 0.1275 ± 0.0647 | Best: 0.2250 | Found: 0.0/20

>>> Running: gcn | degree | weibo | Budget=40
RESULT: (weibo|gcn|degree|B=40) AUC: 0.1796 ± 0.1181 | Best: 0.3588 | Found: 0.0/40

>>> Running: gcn | degree | weibo | Budget=80
RESULT: (weibo|gcn|degree|B=80) AUC: 0.1539 ± 0.0951 | Best: 0.3213 | Found: 0.0/80

>>> Running: gcn | degree | weibo | Budget=160
RESULT: (weibo|gcn|degree|B=160) AUC: 0.4474 ± 0.1201 | Best: 0.6029 | Found: 1.0/160

>>> Running: gcn | degree | weibo | Budget=320
RESULT: (weibo|gcn|degree|B=320) AUC: 0.5473 ± 0.0917 | Best: 0.7356 | Found: 4.0/320

>>> Running: gcn | pagerank | weibo | Budget=10
RESULT: (weibo|gcn|pagerank|B=10) AUC: 0.2460 ± 0.0297 | Best: 0.2978 | Found: 2.0/10

>>> Running: gcn | pagerank | weibo | Budget=20
RESULT: (weibo|gcn|pagerank|B=20) AUC: 0.6443 ± 0.0542 | Best: 0.7339 | Found: 4.0/20

>>> Running: gcn | pagerank | weibo | Budget=40
RESULT: (weibo|gcn|pagerank|B=40) AUC: 0.7065 ± 0.0265 | Best: 0.7635 | Found: 5.0/40

>>> Running: gcn | pagerank | weibo | Budget=80
RESULT: (weibo|gcn|pagerank|B=80) AUC: 0.6710 ± 0.0434 | Best: 0.7495 | Found: 9.0/80

>>> Running: gcn | pagerank | weibo | Budget=160
RESULT: (weibo|gcn|pagerank|B=160) AUC: 0.7090 ± 0.0299 | Best: 0.7765 | Found: 10.0/160

>>> Running: gcn | pagerank | weibo | Budget=320
RESULT: (weibo|gcn|pagerank|B=320) AUC: 0.6798 ± 0.0042 | Best: 0.6851 | Found: 14.0/320

>>> Running: gcn | density | weibo | Budget=10
RESULT: (weibo|gcn|density|B=10) AUC: 0.1622 ± 0.2067 | Best: 0.7792 | Found: 0.1/10

>>> Running: gcn | density | weibo | Budget=20
RESULT: (weibo|gcn|density|B=20) AUC: 0.2241 ± 0.2118 | Best: 0.8179 | Found: 3.8/20

>>> Running: gcn | density | weibo | Budget=40
RESULT: (weibo|gcn|density|B=40) AUC: 0.2187 ± 0.2199 | Best: 0.8688 | Found: 6.0/40

>>> Running: gcn | density | weibo | Budget=80
RESULT: (weibo|gcn|density|B=80) AUC: 0.3299 ± 0.2087 | Best: 0.6757 | Found: 7.2/80

>>> Running: gcn | density | weibo | Budget=160
RESULT: (weibo|gcn|density|B=160) AUC: 0.6892 ± 0.1627 | Best: 0.8499 | Found: 11.4/160

>>> Running: gcn | density | weibo | Budget=320
RESULT: (weibo|gcn|density|B=320) AUC: 0.7876 ± 0.0574 | Best: 0.8660 | Found: 22.3/320

>>> Running: gcn | coreset | weibo | Budget=10
RESULT: (weibo|gcn|coreset|B=10) AUC: 0.3357 ± 0.2156 | Best: 0.6397 | Found: 1.0/10

>>> Running: gcn | coreset | weibo | Budget=20
RESULT: (weibo|gcn|coreset|B=20) AUC: 0.5802 ± 0.1591 | Best: 0.8090 | Found: 4.4/20

>>> Running: gcn | coreset | weibo | Budget=40
RESULT: (weibo|gcn|coreset|B=40) AUC: 0.6492 ± 0.1127 | Best: 0.8286 | Found: 9.4/40

>>> Running: gcn | coreset | weibo | Budget=80
RESULT: (weibo|gcn|coreset|B=80) AUC: 0.7358 ± 0.0312 | Best: 0.7810 | Found: 19.7/80

>>> Running: gcn | coreset | weibo | Budget=160
RESULT: (weibo|gcn|coreset|B=160) AUC: 0.8123 ± 0.0381 | Best: 0.8791 | Found: 39.2/160

>>> Running: gcn | coreset | weibo | Budget=320
RESULT: (weibo|gcn|coreset|B=320) AUC: 0.8720 ± 0.0487 | Best: 0.9411 | Found: 73.6/320

>>> Running: gcn | age | weibo | Budget=10
RESULT: (weibo|gcn|age|B=10) AUC: 0.1414 ± 0.2061 | Best: 0.7586 | Found: 0.1/10

>>> Running: gcn | age | weibo | Budget=20
RESULT: (weibo|gcn|age|B=20) AUC: 0.1319 ± 0.1975 | Best: 0.7240 | Found: 0.3/20

>>> Running: gcn | age | weibo | Budget=40
RESULT: (weibo|gcn|age|B=40) AUC: 0.1604 ± 0.2429 | Best: 0.8845 | Found: 0.6/40

>>> Running: gcn | age | weibo | Budget=80
RESULT: (weibo|gcn|age|B=80) AUC: 0.2288 ± 0.2029 | Best: 0.8076 | Found: 1.2/80

>>> Running: gcn | age | weibo | Budget=160
RESULT: (weibo|gcn|age|B=160) AUC: 0.3435 ± 0.2064 | Best: 0.8059 | Found: 2.3/160

>>> Running: gcn | age | weibo | Budget=320
RESULT: (weibo|gcn|age|B=320) AUC: 0.5100 ± 0.1533 | Best: 0.6527 | Found: 7.0/320

>>> Running: gcn | featprop | weibo | Budget=10
RESULT: (weibo|gcn|featprop|B=10) AUC: 0.5358 ± 0.2118 | Best: 0.7738 | Found: 3.1/10

>>> Running: gcn | featprop | weibo | Budget=20
RESULT: (weibo|gcn|featprop|B=20) AUC: 0.5342 ± 0.0888 | Best: 0.6809 | Found: 6.3/20

>>> Running: gcn | featprop | weibo | Budget=40
RESULT: (weibo|gcn|featprop|B=40) AUC: 0.6398 ± 0.0736 | Best: 0.7140 | Found: 11.4/40

>>> Running: gcn | featprop | weibo | Budget=80
RESULT: (weibo|gcn|featprop|B=80) AUC: 0.7507 ± 0.0311 | Best: 0.7971 | Found: 20.0/80

>>> Running: gcn | featprop | weibo | Budget=160
RESULT: (weibo|gcn|featprop|B=160) AUC: 0.8512 ± 0.0334 | Best: 0.9091 | Found: 41.7/160

>>> Running: gcn | featprop | weibo | Budget=320
RESULT: (weibo|gcn|featprop|B=320) AUC: 0.9027 ± 0.0227 | Best: 0.9336 | Found: 84.2/320

>>> Running: gcn | graphpart | weibo | Budget=10
RESULT: (weibo|gcn|graphpart|B=10) AUC: 0.1329 ± 0.1559 | Best: 0.5979 | Found: 0.1/10

>>> Running: gcn | graphpart | weibo | Budget=20
RESULT: (weibo|gcn|graphpart|B=20) AUC: 0.1915 ± 0.1306 | Best: 0.5821 | Found: 1.2/20

>>> Running: gcn | graphpart | weibo | Budget=40
RESULT: (weibo|gcn|graphpart|B=40) AUC: 0.2410 ± 0.0973 | Best: 0.5266 | Found: 2.6/40

>>> Running: gcn | graphpart | weibo | Budget=80
RESULT: (weibo|gcn|graphpart|B=80) AUC: 0.5529 ± 0.1838 | Best: 0.7478 | Found: 5.9/80

>>> Running: gcn | graphpart | weibo | Budget=160
RESULT: (weibo|gcn|graphpart|B=160) AUC: 0.7117 ± 0.0720 | Best: 0.8672 | Found: 14.3/160

>>> Running: gcn | graphpart | weibo | Budget=320
RESULT: (weibo|gcn|graphpart|B=320) AUC: 0.8000 ± 0.0546 | Best: 0.8690 | Found: 26.8/320

>>> Running: gcn | graphpartfar | weibo | Budget=10
RESULT: (weibo|gcn|graphpartfar|B=10) AUC: 0.1329 ± 0.1559 | Best: 0.5979 | Found: 0.1/10

>>> Running: gcn | graphpartfar | weibo | Budget=20
RESULT: (weibo|gcn|graphpartfar|B=20) AUC: 0.1468 ± 0.1608 | Best: 0.5981 | Found: 0.4/20

>>> Running: gcn | graphpartfar | weibo | Budget=40
RESULT: (weibo|gcn|graphpartfar|B=40) AUC: 0.4124 ± 0.2256 | Best: 0.8719 | Found: 2.7/40

>>> Running: gcn | graphpartfar | weibo | Budget=80
RESULT: (weibo|gcn|graphpartfar|B=80) AUC: 0.4999 ± 0.1714 | Best: 0.8084 | Found: 6.0/80

>>> Running: gcn | graphpartfar | weibo | Budget=160
RESULT: (weibo|gcn|graphpartfar|B=160) AUC: 0.6873 ± 0.0716 | Best: 0.7739 | Found: 12.7/160

>>> Running: gcn | graphpartfar | weibo | Budget=320
RESULT: (weibo|gcn|graphpartfar|B=320) AUC: 0.7594 ± 0.0628 | Best: 0.8192 | Found: 26.1/320


------------


>>> Running: sage | random | weibo | Budget=10
RESULT: (weibo|sage|random|B=10) AUC: 0.2887 ± 0.1096 | Best: 0.5395 | Found: 0.3/10

>>> Running: sage | random | weibo | Budget=20
RESULT: (weibo|sage|random|B=20) AUC: 0.2395 ± 0.0575 | Best: 0.3245 | Found: 0.5/20

>>> Running: sage | random | weibo | Budget=40
RESULT: (weibo|sage|random|B=40) AUC: 0.2268 ± 0.0571 | Best: 0.3364 | Found: 0.9/40

>>> Running: sage | random | weibo | Budget=80
RESULT: (weibo|sage|random|B=80) AUC: 0.3805 ± 0.1337 | Best: 0.6344 | Found: 3.1/80

>>> Running: sage | random | weibo | Budget=160
RESULT: (weibo|sage|random|B=160) AUC: 0.4542 ± 0.0939 | Best: 0.5877 | Found: 6.4/160

>>> Running: sage | random | weibo | Budget=320
RESULT: (weibo|sage|random|B=320) AUC: 0.5938 ± 0.0780 | Best: 0.7385 | Found: 13.6/320

>>> Running: sage | uncertainty | weibo | Budget=10
RESULT: (weibo|sage|uncertainty|B=10) AUC: 0.3772 ± 0.0862 | Best: 0.4896 | Found: 3.3/10

>>> Running: sage | uncertainty | weibo | Budget=20
RESULT: (weibo|sage|uncertainty|B=20) AUC: 0.3674 ± 0.1232 | Best: 0.6085 | Found: 4.7/20

>>> Running: sage | uncertainty | weibo | Budget=40
RESULT: (weibo|sage|uncertainty|B=40) AUC: 0.3839 ± 0.1285 | Best: 0.6372 | Found: 9.5/40

>>> Running: sage | uncertainty | weibo | Budget=80
RESULT: (weibo|sage|uncertainty|B=80) AUC: 0.3901 ± 0.0943 | Best: 0.5750 | Found: 16.8/80

>>> Running: sage | uncertainty | weibo | Budget=160
RESULT: (weibo|sage|uncertainty|B=160) AUC: 0.3667 ± 0.1187 | Best: 0.6423 | Found: 25.7/160

>>> Running: sage | uncertainty | weibo | Budget=320
RESULT: (weibo|sage|uncertainty|B=320) AUC: 0.4517 ± 0.1269 | Best: 0.6797 | Found: 37.1/320

>>> Running: sage | degree | weibo | Budget=10
RESULT: (weibo|sage|degree|B=10) AUC: 0.2806 ± 0.0805 | Best: 0.4059 | Found: 0.0/10

>>> Running: sage | degree | weibo | Budget=20
RESULT: (weibo|sage|degree|B=20) AUC: 0.2658 ± 0.0826 | Best: 0.4091 | Found: 0.0/20

>>> Running: sage | degree | weibo | Budget=40
RESULT: (weibo|sage|degree|B=40) AUC: 0.2721 ± 0.0791 | Best: 0.3857 | Found: 0.0/40

>>> Running: sage | degree | weibo | Budget=80
RESULT: (weibo|sage|degree|B=80) AUC: 0.2761 ± 0.0714 | Best: 0.3694 | Found: 0.0/80

>>> Running: sage | degree | weibo | Budget=160
RESULT: (weibo|sage|degree|B=160) AUC: 0.2637 ± 0.0561 | Best: 0.3626 | Found: 1.0/160

>>> Running: sage | degree | weibo | Budget=320
RESULT: (weibo|sage|degree|B=320) AUC: 0.3383 ± 0.0689 | Best: 0.4930 | Found: 4.0/320

>>> Running: sage | pagerank | weibo | Budget=10
RESULT: (weibo|sage|pagerank|B=10) AUC: 0.2524 ± 0.0229 | Best: 0.2888 | Found: 2.0/10

>>> Running: sage | pagerank | weibo | Budget=20
RESULT: (weibo|sage|pagerank|B=20) AUC: 0.4594 ± 0.0694 | Best: 0.5775 | Found: 4.0/20

>>> Running: sage | pagerank | weibo | Budget=40
RESULT: (weibo|sage|pagerank|B=40) AUC: 0.4775 ± 0.0510 | Best: 0.5819 | Found: 5.0/40

>>> Running: sage | pagerank | weibo | Budget=80
RESULT: (weibo|sage|pagerank|B=80) AUC: 0.6535 ± 0.0313 | Best: 0.7038 | Found: 9.0/80

>>> Running: sage | pagerank | weibo | Budget=160
RESULT: (weibo|sage|pagerank|B=160) AUC: 0.6191 ± 0.0322 | Best: 0.6673 | Found: 10.0/160

>>> Running: sage | pagerank | weibo | Budget=320
RESULT: (weibo|sage|pagerank|B=320) AUC: 0.5830 ± 0.0125 | Best: 0.6045 | Found: 14.0/320

>>> Running: sage | density | weibo | Budget=10
RESULT: (weibo|sage|density|B=10) AUC: 0.2655 ± 0.0643 | Best: 0.3594 | Found: 0.1/10

>>> Running: sage | density | weibo | Budget=20
RESULT: (weibo|sage|density|B=20) AUC: 0.2724 ± 0.0559 | Best: 0.3545 | Found: 0.4/20

>>> Running: sage | density | weibo | Budget=40
RESULT: (weibo|sage|density|B=40) AUC: 0.3146 ± 0.1227 | Best: 0.5654 | Found: 1.5/40

>>> Running: sage | density | weibo | Budget=80
RESULT: (weibo|sage|density|B=80) AUC: 0.3336 ± 0.0950 | Best: 0.4707 | Found: 5.2/80

>>> Running: sage | density | weibo | Budget=160
RESULT: (weibo|sage|density|B=160) AUC: 0.3485 ± 0.1024 | Best: 0.5097 | Found: 11.0/160

>>> Running: sage | density | weibo | Budget=320
RESULT: (weibo|sage|density|B=320) AUC: 0.4940 ± 0.1464 | Best: 0.6959 | Found: 20.6/320

>>> Running: sage | coreset | weibo | Budget=10
RESULT: (weibo|sage|coreset|B=10) AUC: 0.4505 ± 0.2035 | Best: 0.8264 | Found: 2.1/10

>>> Running: sage | coreset | weibo | Budget=20
RESULT: (weibo|sage|coreset|B=20) AUC: 0.5014 ± 0.1487 | Best: 0.7716 | Found: 4.7/20

>>> Running: sage | coreset | weibo | Budget=40
RESULT: (weibo|sage|coreset|B=40) AUC: 0.4530 ± 0.1446 | Best: 0.7181 | Found: 9.1/40

>>> Running: sage | coreset | weibo | Budget=80
RESULT: (weibo|sage|coreset|B=80) AUC: 0.5410 ± 0.0859 | Best: 0.6617 | Found: 15.1/80

>>> Running: sage | coreset | weibo | Budget=160
RESULT: (weibo|sage|coreset|B=160) AUC: 0.6430 ± 0.0501 | Best: 0.7497 | Found: 30.8/160

>>> Running: sage | coreset | weibo | Budget=320
RESULT: (weibo|sage|coreset|B=320) AUC: 0.7113 ± 0.0463 | Best: 0.7768 | Found: 59.5/320

>>> Running: sage | age | weibo | Budget=10
RESULT: (weibo|sage|age|B=10) AUC: 0.2870 ± 0.0560 | Best: 0.3651 | Found: 0.2/10

>>> Running: sage | age | weibo | Budget=20
RESULT: (weibo|sage|age|B=20) AUC: 0.2160 ± 0.0551 | Best: 0.3432 | Found: 0.3/20

>>> Running: sage | age | weibo | Budget=40
RESULT: (weibo|sage|age|B=40) AUC: 0.2292 ± 0.1417 | Best: 0.6420 | Found: 0.7/40

>>> Running: sage | age | weibo | Budget=80
RESULT: (weibo|sage|age|B=80) AUC: 0.2878 ± 0.0949 | Best: 0.5125 | Found: 1.4/80

>>> Running: sage | age | weibo | Budget=160
RESULT: (weibo|sage|age|B=160) AUC: 0.3341 ± 0.1069 | Best: 0.5225 | Found: 5.1/160

>>> Running: sage | age | weibo | Budget=320
RESULT: (weibo|sage|age|B=320) AUC: 0.3843 ± 0.1072 | Best: 0.4979 | Found: 8.6/320

>>> Running: sage | featprop | weibo | Budget=10
RESULT: (weibo|sage|featprop|B=10) AUC: 0.4583 ± 0.1980 | Best: 0.7386 | Found: 3.1/10

>>> Running: sage | featprop | weibo | Budget=20
RESULT: (weibo|sage|featprop|B=20) AUC: 0.4700 ± 0.1148 | Best: 0.7016 | Found: 6.3/20

>>> Running: sage | featprop | weibo | Budget=40
RESULT: (weibo|sage|featprop|B=40) AUC: 0.5364 ± 0.1025 | Best: 0.6785 | Found: 11.4/40

>>> Running: sage | featprop | weibo | Budget=80
RESULT: (weibo|sage|featprop|B=80) AUC: 0.4853 ± 0.0260 | Best: 0.5349 | Found: 20.0/80

>>> Running: sage | featprop | weibo | Budget=160
RESULT: (weibo|sage|featprop|B=160) AUC: 0.6220 ± 0.0426 | Best: 0.7134 | Found: 41.7/160

>>> Running: sage | featprop | weibo | Budget=320
RESULT: (weibo|sage|featprop|B=320) AUC: 0.7467 ± 0.0302 | Best: 0.8062 | Found: 84.2/320

>>> Running: sage | graphpart | weibo | Budget=10
RESULT: (weibo|sage|graphpart|B=10) AUC: 0.1879 ± 0.0442 | Best: 0.3042 | Found: 0.1/10

>>> Running: sage | graphpart | weibo | Budget=20
RESULT: (weibo|sage|graphpart|B=20) AUC: 0.1674 ± 0.0451 | Best: 0.2395 | Found: 1.2/20

>>> Running: sage | graphpart | weibo | Budget=40
RESULT: (weibo|sage|graphpart|B=40) AUC: 0.2617 ± 0.0603 | Best: 0.4041 | Found: 2.6/40

>>> Running: sage | graphpart | weibo | Budget=80
RESULT: (weibo|sage|graphpart|B=80) AUC: 0.3867 ± 0.0972 | Best: 0.5565 | Found: 5.9/80

>>> Running: sage | graphpart | weibo | Budget=160
RESULT: (weibo|sage|graphpart|B=160) AUC: 0.4767 ± 0.0800 | Best: 0.6281 | Found: 14.3/160

>>> Running: sage | graphpart | weibo | Budget=320
RESULT: (weibo|sage|graphpart|B=320) AUC: 0.6349 ± 0.0507 | Best: 0.7408 | Found: 26.8/320

>>> Running: sage | graphpartfar | weibo | Budget=10
RESULT: (weibo|sage|graphpartfar|B=10) AUC: 0.1879 ± 0.0442 | Best: 0.3042 | Found: 0.1/10

>>> Running: sage | graphpartfar | weibo | Budget=20
RESULT: (weibo|sage|graphpartfar|B=20) AUC: 0.1591 ± 0.0822 | Best: 0.3150 | Found: 0.4/20

>>> Running: sage | graphpartfar | weibo | Budget=40
RESULT: (weibo|sage|graphpartfar|B=40) AUC: 0.3029 ± 0.1729 | Best: 0.7505 | Found: 2.7/40

>>> Running: sage | graphpartfar | weibo | Budget=80
RESULT: (weibo|sage|graphpartfar|B=80) AUC: 0.4156 ± 0.1275 | Best: 0.6650 | Found: 6.0/80

>>> Running: sage | graphpartfar | weibo | Budget=160
RESULT: (weibo|sage|graphpartfar|B=160) AUC: 0.5064 ± 0.0940 | Best: 0.6384 | Found: 12.7/160

>>> Running: sage | graphpartfar | weibo | Budget=320
RESULT: (weibo|sage|graphpartfar|B=320) AUC: 0.6074 ± 0.0618 | Best: 0.7435 | Found: 26.1/320


----------


>>> Running: gat | random | weibo | Budget=10
RESULT: (weibo|gat|random|B=10) AUC: 0.5501 ± 0.1505 | Best: 0.7928 | Found: 0.3/10

>>> Running: gat | random | weibo | Budget=20
RESULT: (weibo|gat|random|B=20) AUC: 0.5087 ± 0.1356 | Best: 0.7217 | Found: 0.5/20

>>> Running: gat | random | weibo | Budget=40
RESULT: (weibo|gat|random|B=40) AUC: 0.5387 ± 0.1665 | Best: 0.8769 | Found: 0.9/40

>>> Running: gat | random | weibo | Budget=80
RESULT: (weibo|gat|random|B=80) AUC: 0.6464 ± 0.1384 | Best: 0.8845 | Found: 3.1/80

>>> Running: gat | random | weibo | Budget=160
RESULT: (weibo|gat|random|B=160) AUC: 0.7213 ± 0.1238 | Best: 0.9161 | Found: 6.4/160

>>> Running: gat | random | weibo | Budget=320
RESULT: (weibo|gat|random|B=320) AUC: 0.7459 ± 0.0992 | Best: 0.8848 | Found: 13.6/320

>>> Running: gat | uncertainty | weibo | Budget=10
RESULT: (weibo|gat|uncertainty|B=10) AUC: 0.6065 ± 0.1816 | Best: 0.8557 | Found: 1.3/10

>>> Running: gat | uncertainty | weibo | Budget=20
RESULT: (weibo|gat|uncertainty|B=20) AUC: 0.5594 ± 0.1455 | Best: 0.7787 | Found: 3.9/20

>>> Running: gat | uncertainty | weibo | Budget=40
RESULT: (weibo|gat|uncertainty|B=40) AUC: 0.5750 ± 0.1271 | Best: 0.8274 | Found: 6.5/40

>>> Running: gat | uncertainty | weibo | Budget=80
RESULT: (weibo|gat|uncertainty|B=80) AUC: 0.5912 ± 0.1608 | Best: 0.8371 | Found: 13.4/80

>>> Running: gat | uncertainty | weibo | Budget=160
RESULT: (weibo|gat|uncertainty|B=160) AUC: 0.6721 ± 0.1346 | Best: 0.9052 | Found: 24.3/160

>>> Running: gat | uncertainty | weibo | Budget=320
RESULT: (weibo|gat|uncertainty|B=320) AUC: 0.7123 ± 0.1128 | Best: 0.8984 | Found: 46.8/320

>>> Running: gat | degree | weibo | Budget=10
RESULT: (weibo|gat|degree|B=10) AUC: 0.5726 ± 0.1423 | Best: 0.8668 | Found: 0.0/10

>>> Running: gat | degree | weibo | Budget=20
RESULT: (weibo|gat|degree|B=20) AUC: 0.5094 ± 0.1593 | Best: 0.8542 | Found: 0.0/20

>>> Running: gat | degree | weibo | Budget=40
RESULT: (weibo|gat|degree|B=40) AUC: 0.5007 ± 0.1516 | Best: 0.7972 | Found: 0.0/40

>>> Running: gat | degree | weibo | Budget=80
RESULT: (weibo|gat|degree|B=80) AUC: 0.5433 ± 0.1636 | Best: 0.8203 | Found: 0.0/80

>>> Running: gat | degree | weibo | Budget=160
RESULT: (weibo|gat|degree|B=160) AUC: 0.4999 ± 0.1711 | Best: 0.7098 | Found: 1.0/160

>>> Running: gat | degree | weibo | Budget=320
RESULT: (weibo|gat|degree|B=320) AUC: 0.6094 ± 0.1628 | Best: 0.8281 | Found: 4.0/320

>>> Running: gat | pagerank | weibo | Budget=10
RESULT: (weibo|gat|pagerank|B=10) AUC: 0.5408 ± 0.1429 | Best: 0.8099 | Found: 2.0/10

>>> Running: gat | pagerank | weibo | Budget=20
RESULT: (weibo|gat|pagerank|B=20) AUC: 0.6745 ± 0.1328 | Best: 0.8199 | Found: 4.0/20

>>> Running: gat | pagerank | weibo | Budget=40
RESULT: (weibo|gat|pagerank|B=40) AUC: 0.6920 ± 0.1309 | Best: 0.8163 | Found: 5.0/40

>>> Running: gat | pagerank | weibo | Budget=80
RESULT: (weibo|gat|pagerank|B=80) AUC: 0.7445 ± 0.0791 | Best: 0.8180 | Found: 9.0/80

>>> Running: gat | pagerank | weibo | Budget=160
RESULT: (weibo|gat|pagerank|B=160) AUC: 0.7355 ± 0.0818 | Best: 0.8091 | Found: 10.0/160

>>> Running: gat | pagerank | weibo | Budget=320
RESULT: (weibo|gat|pagerank|B=320) AUC: 0.7469 ± 0.1008 | Best: 0.8730 | Found: 14.0/320

>>> Running: gat | density | weibo | Budget=10
RESULT: (weibo|gat|density|B=10) AUC: 0.5031 ± 0.1757 | Best: 0.7831 | Found: 0.1/10

>>> Running: gat | density | weibo | Budget=20
RESULT: (weibo|gat|density|B=20) AUC: 0.5394 ± 0.1846 | Best: 0.9129 | Found: 0.2/20

>>> Running: gat | density | weibo | Budget=40
RESULT: (weibo|gat|density|B=40) AUC: 0.5300 ± 0.1508 | Best: 0.8022 | Found: 0.6/40

>>> Running: gat | density | weibo | Budget=80
RESULT: (weibo|gat|density|B=80) AUC: 0.5850 ± 0.1725 | Best: 0.8632 | Found: 1.1/80

>>> Running: gat | density | weibo | Budget=160
RESULT: (weibo|gat|density|B=160) AUC: 0.6811 ± 0.1381 | Best: 0.9055 | Found: 4.4/160

>>> Running: gat | density | weibo | Budget=320
RESULT: (weibo|gat|density|B=320) AUC: 0.7171 ± 0.1324 | Best: 0.9344 | Found: 7.3/320

>>> Running: gat | coreset | weibo | Budget=10
RESULT: (weibo|gat|coreset|B=10) AUC: 0.5529 ± 0.1342 | Best: 0.8049 | Found: 1.7/10

>>> Running: gat | coreset | weibo | Budget=20
RESULT: (weibo|gat|coreset|B=20) AUC: 0.6154 ± 0.1751 | Best: 0.8702 | Found: 4.0/20

>>> Running: gat | coreset | weibo | Budget=40
RESULT: (weibo|gat|coreset|B=40) AUC: 0.6365 ± 0.1499 | Best: 0.8687 | Found: 6.6/40

>>> Running: gat | coreset | weibo | Budget=80
RESULT: (weibo|gat|coreset|B=80) AUC: 0.6356 ± 0.1229 | Best: 0.8604 | Found: 15.6/80

>>> Running: gat | coreset | weibo | Budget=160
RESULT: (weibo|gat|coreset|B=160) AUC: 0.7138 ± 0.0711 | Best: 0.8257 | Found: 26.6/160

>>> Running: gat | coreset | weibo | Budget=320
RESULT: (weibo|gat|coreset|B=320) AUC: 0.7811 ± 0.0753 | Best: 0.8775 | Found: 53.9/320

>>> Running: gat | age | weibo | Budget=10
RESULT: (weibo|gat|age|B=10) AUC: 0.5511 ± 0.1762 | Best: 0.8642 | Found: 0.3/10

>>> Running: gat | age | weibo | Budget=20
RESULT: (weibo|gat|age|B=20) AUC: 0.5434 ± 0.1849 | Best: 0.8535 | Found: 0.5/20

>>> Running: gat | age | weibo | Budget=40
RESULT: (weibo|gat|age|B=40) AUC: 0.5418 ± 0.1573 | Best: 0.8866 | Found: 1.0/40

>>> Running: gat | age | weibo | Budget=80
RESULT: (weibo|gat|age|B=80) AUC: 0.5545 ± 0.1321 | Best: 0.7226 | Found: 2.6/80

>>> Running: gat | age | weibo | Budget=160
RESULT: (weibo|gat|age|B=160) AUC: 0.6878 ± 0.1370 | Best: 0.9081 | Found: 5.6/160

>>> Running: gat | age | weibo | Budget=320
RESULT: (weibo|gat|age|B=320) AUC: 0.6928 ± 0.1317 | Best: 0.9181 | Found: 12.0/320

>>> Running: gat | featprop | weibo | Budget=10
RESULT: (weibo|gat|featprop|B=10) AUC: 0.4547 ± 0.1926 | Best: 0.8559 | Found: 3.1/10

>>> Running: gat | featprop | weibo | Budget=20
RESULT: (weibo|gat|featprop|B=20) AUC: 0.6177 ± 0.1925 | Best: 0.9320 | Found: 6.3/20

>>> Running: gat | featprop | weibo | Budget=40
RESULT: (weibo|gat|featprop|B=40) AUC: 0.6477 ± 0.1205 | Best: 0.8172 | Found: 11.4/40

>>> Running: gat | featprop | weibo | Budget=80
RESULT: (weibo|gat|featprop|B=80) AUC: 0.6891 ± 0.0585 | Best: 0.8122 | Found: 20.0/80

>>> Running: gat | featprop | weibo | Budget=160
RESULT: (weibo|gat|featprop|B=160) AUC: 0.7792 ± 0.0736 | Best: 0.8845 | Found: 41.7/160

>>> Running: gat | featprop | weibo | Budget=320
RESULT: (weibo|gat|featprop|B=320) AUC: 0.8462 ± 0.0395 | Best: 0.9047 | Found: 84.2/320

>>> Running: gat | graphpart | weibo | Budget=10
RESULT: (weibo|gat|graphpart|B=10) AUC: 0.5235 ± 0.1131 | Best: 0.7370 | Found: 0.1/10

>>> Running: gat | graphpart | weibo | Budget=20
RESULT: (weibo|gat|graphpart|B=20) AUC: 0.4541 ± 0.1606 | Best: 0.7431 | Found: 1.2/20

>>> Running: gat | graphpart | weibo | Budget=40
RESULT: (weibo|gat|graphpart|B=40) AUC: 0.5330 ± 0.1614 | Best: 0.7565 | Found: 2.6/40

>>> Running: gat | graphpart | weibo | Budget=80
RESULT: (weibo|gat|graphpart|B=80) AUC: 0.6037 ± 0.2026 | Best: 0.9320 | Found: 5.9/80

>>> Running: gat | graphpart | weibo | Budget=160
RESULT: (weibo|gat|graphpart|B=160) AUC: 0.6434 ± 0.1258 | Best: 0.9377 | Found: 14.3/160

>>> Running: gat | graphpart | weibo | Budget=320
RESULT: (weibo|gat|graphpart|B=320) AUC: 0.7548 ± 0.0747 | Best: 0.8425 | Found: 26.8/320

>>> Running: gat | graphpartfar | weibo | Budget=10
RESULT: (weibo|gat|graphpartfar|B=10) AUC: 0.5273 ± 0.1150 | Best: 0.7434 | Found: 0.1/10

>>> Running: gat | graphpartfar | weibo | Budget=20
RESULT: (weibo|gat|graphpartfar|B=20) AUC: 0.5055 ± 0.1429 | Best: 0.6782 | Found: 0.4/20

>>> Running: gat | graphpartfar | weibo | Budget=40
RESULT: (weibo|gat|graphpartfar|B=40) AUC: 0.5269 ± 0.1467 | Best: 0.7260 | Found: 2.7/40

>>> Running: gat | graphpartfar | weibo | Budget=80
RESULT: (weibo|gat|graphpartfar|B=80) AUC: 0.6893 ± 0.0982 | Best: 0.8529 | Found: 6.0/80

>>> Running: gat | graphpartfar | weibo | Budget=160
RESULT: (weibo|gat|graphpartfar|B=160) AUC: 0.6865 ± 0.1404 | Best: 0.9161 | Found: 12.7/160

>>> Running: gat | graphpartfar | weibo | Budget=320
RESULT: (weibo|gat|graphpartfar|B=320) AUC: 0.7181 ± 0.1270 | Best: 0.8961 | Found: 26.1/320

"""

# ======================================================
# 2. 自動處理邏輯
# ======================================================

def parse_log_data(data_string):
    """ 解析 Log 資料字串，回傳結構化字典 """
    # 儲存結構: results[model][method][budget] = {mean, std, best, found}
    results = collections.defaultdict(
        lambda: collections.defaultdict(
            lambda: collections.defaultdict(dict)
        )
    )
    
    lines = data_string.strip().split('\n')
    # Regex 解析: RESULT: (...) AUC: mean ± std | Best: best | Found: found/total
    # 注意: Found 後面只抓取數值部分 (例如 0.3)，不包含分母 /10
    pattern = re.compile(r"RESULT: \((.*?)\|(.*?)\|(.*?)\|B=(\d+)\) AUC: ([\d\.]+) ± ([\d\.]+) \| Best: ([\d\.]+) \| Found: ([\d\.]+)")

    for line in lines:
        line = line.strip()
        if not line.startswith("RESULT:"):
            continue
            
        match = pattern.search(line)
        if match:
            model = match.group(2)
            method = match.group(3)
            budget = int(match.group(4))
            
            results[model][method][budget] = {
                'mean': float(match.group(5)),
                'std': float(match.group(6)),
                'best': float(match.group(7)),
                'found': float(match.group(8))
            }
            
    return results

def get_method_display_name(method):
    """ 將方法名稱轉為論文顯示格式 """
    mapping = {
        'random': 'Random',
        'uncertainty': 'Uncertainty',
        'degree': 'Degree',
        'pagerank': 'PageRank',
        'density': 'Density',
        'coreset': 'CoreSet',
        'age': 'AGE',
        'featprop': 'FeatProp',
        'graphpart': 'GraphPart',
        'graphpartfar': 'GraphPartFar'
    }
    return mapping.get(method, method.capitalize())

def generate_latex_table(results):
    """ 生成 LaTeX 表格代碼 """
    
    # 定義顯示順序
    methods_order = [
        'random', 'uncertainty', 'degree', 'pagerank', 'density', 
        'coreset', 'age', 'featprop', 'graphpart', 'graphpartfar'
    ]
    budgets_order = [10, 20, 40, 80, 160, 320]
    models_order = ['gcn', 'sage', 'gat']

    output = []
    
    # 加入使用說明註解
    output.append(r"% ===========================================")
    output.append(r"% Paste this code into your Overleaf document.")
    output.append(r"% Required Packages: \usepackage{booktabs}, \usepackage{graphicx}")
    output.append(r"% ===========================================")
    output.append("")

    for model in models_order:
        if model not in results:
            continue
            
        # --- 計算排名邏輯 ---
        rankings = {}
        for b in budgets_order:
            scores = []
            for m in methods_order:
                if m in results[model] and b in results[model][m]:
                    scores.append( (results[model][m][b]['mean'], m) )
            
            # 排序：分數高到低
            scores.sort(key=lambda x: x[0], reverse=True)
            
            rankings[b] = {
                'first': scores[0][1] if len(scores) > 0 else None,
                'second': scores[1][1] if len(scores) > 1 else None
            }

        # --- 開始寫入 LaTeX 表格 ---
        output.append(r"\begin{table*}[ht]")
        output.append(r"\centering")
        output.append(r"\caption{Performance Comparison on Weibo using " + model.upper() + r" (Metric: ROC-AUC).}")
        output.append(r"\resizebox{\textwidth}{!}{")
        output.append(r"\begin{tabular}{l" + "c" * len(budgets_order) + r"}")
        output.append(r"\toprule")
        
        # 表頭 (Budgets)
        header = "Method & " + " & ".join([f"B={b}" for b in budgets_order]) + r" \\"
        output.append(header)
        output.append(r"\midrule")
        
        # 內容行 (Methods)
        for m in methods_order:
            row_str = get_method_display_name(m)
            
            for b in budgets_order:
                if m in results[model] and b in results[model][m]:
                    data = results[model][m][b]
                    mean, std = data['mean'], data['std']
                    best, found = data['best'], data['found']
                    
                    # 1. 格式化 AUC 數值
                    auc_text = f"{mean:.4f} $\pm$ {std:.4f}"
                    
                    # 2. 判斷排名並加粗/底線
                    if rankings[b]['first'] == m:
                        auc_text = r"\textbf{" + auc_text + r"}"
                    elif rankings[b]['second'] == m:
                        auc_text = r"\underline{" + auc_text + r"}"
                    
                    # 3. 組合副數據 (Best & Found)
                    # 使用巢狀 tabular 來實現單元格內換行，這是最穩定的方式
                    info_text = f"B: {best:.4f} | F: {found:.1f}"
                    cell_content = r"\begin{tabular}{@{}c@{}}" + auc_text + r" \\ \tiny{" + info_text + r"}\end{tabular}"
                    
                    row_str += " & " + cell_content
                else:
                    row_str += " & -"
            
            row_str += r" \\"
            output.append(row_str)
            # 加一點行距讓表格不要太擁擠
            output.append(r"\addlinespace[0.2em]")

        output.append(r"\bottomrule")
        output.append(r"\end{tabular}}")
        output.append(r"\label{tab:res_" + model + r"}")
        output.append(r"\end{table*}")
        output.append("") # 空行分隔不同模型

    return "\n".join(output)

# ======================================================
# 3. 執行程式
# ======================================================

if __name__ == "__main__":
    if len(raw_data.strip()) < 10:
        print("警告: raw_data 似乎是空的，請先將你的 Log 貼入程式碼中的 raw_data 變數中！")
    else:
        parsed_results = parse_log_data(raw_data)
        latex_code = generate_latex_table(parsed_results)
        print(latex_code)