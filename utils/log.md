## 为什么效果有差距？

- 数据预处理部分有问题，没有筛选用户。  // 筛选出部分有用户后， 依然效果较差。

## 先测试union_g单独的效果

    trust_rate = 1     1.0665
    2
    3     1.0692
                 4     1.0717
                 5     1.0749
                 ...

    r-r  10    1.08


## ...直接加上好像效果不是很好， 同时要调整r2变得更小 0.0001才能训练

## 初始化参数

    多了一个没有用的 egat层，却对模型效果产生了提升， 网上说是因为多了参数，那么其他参数的初始化会发生变化。

## Modify:

    r1 r2, r3: 0.0001, 0.001, 0.01
    Train_MSE:0.7273 Test_rmse:1.1150 Test_mae: 0.8473 at epoch 32

### Modify:   simple

   r1 r2, r3: 0.0001, 0.001, 0.01  test_MSE:1.0665063858032227    test_mae: 0.8065751791000366
   Train_Loss:0.6387327943049687   test_MSE:1.0665063858032227    test_mae: 0.8065751791000366

### Modify:   k = 0.5

   r1 r2, r3: 0.0001, 0.001, 0.01  test_MSE:1.1170070171356201    test_mae: 0.8941329717636108
   Train_Loss:0.6214676377638965   test_MSE:1.1170070171356201    test_mae: 0.8941329717636108

### Modify:   k = 0.9

   r1 r2, r3: 0.0001, 0.001, 0.01  test_MSE:1.0728440284729004    test_mae: 0.8259971737861633
   Train_Loss:0.597546185406161   test_MSE:1.0728440284729004    test_mae: 0.8259971737861633

### Modify:  

   r1 r2, r3: 0.0001, 0.001, 0.01
   Train_Loss:0.5641083818086436   test_MSE:1.097679615020752    test_mae: 0.8372775316238403

### Modify:

   r1 r2, r3: 0.0001, 0.001, 0.01
   Train_Loss:0.5922929883842737   test_MSE:1.0752079486846924    test_mae: 0.8244116306304932

# Modify:  self = 10  rate_g
   r1 r2, r3: 0.0001, 0.001, 0.01
   Train_Loss:0.6101732270818361   test_MSE:1.083341121673584    test_mae: 0.8233314752578735

# Modify:  self = 8  rate_g
   r1 r2, r3: 0.0001, 0.001, 0.01
   Train_Loss:0.6126291298530471   test_MSE:1.0689826011657715    test_mae: 0.8149816393852234

# Modify:   self = 7 rate_g
   r1 r2, r3: 0.0001, 0.001, 0.01 
   Train_Loss:0.6205188867071985   test_MSE:1.065439224243164    test_mae: 0.8143481016159058  

# Modify:  self = 6  rate_g
   r1 r2, r3: 0.0001, 0.001, 0.01
   Train_Loss:0.6083245084319316   test_MSE:1.071096658706665    test_mae: 0.8181809782981873
# Modify:   self = 7 union_g  trust = 1
   r1 r2, r3: 0.0001, 0.001, 0.01 
   Train_Loss:0.6792041415899572   test_MSE:1.0968266725540161    test_mae: 0.8431850671768188
# Modify:   self = 7 union_g  trust = 2  
   r1 r2, r3: 0.0001, 0.001, 0.01 
   Train_Loss:0.6312616522043524   test_MSE:1.0773487091064453    test_mae: 0.8237356543540955
# Modify:    self = 7 union_g  trust = 3
   r1 r2, r3: 0.0001, 0.001, 0.01 
   Train_Loss:0.6150794293678982   test_MSE:1.0656030178070068    test_mae: 0.8118655681610107
# Modify:   self = 7 union_g  trust = 4
   r1 r2, r3: 0.0001, 0.001, 0.01 
   Train_Loss:0.6532642329242867   test_MSE:1.063882827758789    test_mae: 0.8068234324455261   这里可以考虑训练100 epoch
# Modify:   self = 7 union_g  trust = 5
   r1 r2, r3: 0.0001, 0.001, 0.01 
   Train_Loss:0.6531387139374102   test_MSE:1.0734729766845703    test_mae: 0.8217270970344543
# Modify:   self = 7 union_g  trust = 6
   r1 r2, r3: 0.0001, 0.001, 0.01 
   Train_Loss:0.6215026294681388   test_MSE:1.0751187801361084    test_mae: 0.8245963454246521
# Modify:   self = 7 union_g  trust = 4.5
   r1 r2, r3: 0.0001, 0.001, 0.01 
   Train_Loss:0.6432094842615262   test_MSE:1.088836431503296    test_mae: 0.8379480242729187
# Modify:   self = 7 union_g  trust = 3.5
   r1 r2, r3: 0.0001, 0.001, 0.01 
   Train_Loss:0.6306485516924254   test_MSE:1.0634639263153076    test_mae: 0.8088275790214539
# Modify:   
   r1 r2, r3: 0.0001, 0.001, 0.01 
   Train_Loss:0.6143959068076711   test_MSE:1.063882827758789    test_mae: 0.8068234324455261
# Modify:   
   r1 r2, r3: 0.0001, 0.001, 0.01 
   Train_Loss:0.5843731693818536   test_MSE:1.0741522312164307    test_mae: 0.8165513277053833
# Modify:   
   r1 r2, r3: 0.0001, 0.01, 0.001 
   Train_Loss:0.741051864456123   test_MSE:1.0975492000579834    test_mae: 0.839773952960968
# Modify:   
   r1 r2, r3: 0.0001, 0.001, 0.01 
   Train_Loss:0.5843731693818536   test_MSE:1.0741522312164307    test_mae: 0.8165513277053833
# Modify:   
   r1 r2, r3: 0.0001, 0.001, 0.01 
   Train_Loss:0.5734716250862874   test_MSE:1.0747854709625244    test_mae: 0.8127074241638184
# Modify:   
   r1 r2, r3: 0.0001, 0.01, 0.01 
   Train_Loss:0.6377785180656004   test_MSE:1.0669423341751099    test_mae: 0.8189769387245178
# Modify:   
   r1 r2, r3: 0.0001, 0.01, 0.01 
   Train_Loss:0.6069480877527049   test_MSE:1.0614269971847534    test_mae: 0.8152615427970886
# Modify:   
   r1 r2, r3: 0.001, 0.001, 0.001 
   Train_Loss:0.6134756685982288   test_MSE:1.0632458925247192    test_mae: 0.8090497255325317
# Modify:    天哪。。。。。
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:0.6025794316345537   test_MSE:1.037166714668274    test_mae: 0.7984960079193115
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:0.6004574479351581   test_MSE:1.0414185523986816    test_mae: 0.8037333488464355
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:0.6471164872948553   test_MSE:1.0832347869873047    test_mae: 0.835074245929718
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:0.6914028416217213   test_MSE:1.0832347869873047    test_mae: 0.835074245929718
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:0.6697539502466229   test_MSE:1.0847406387329102    test_mae: 0.8360196948051453
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:0.6493770732006556   test_MSE:1.0383634567260742    test_mae: 0.7989957332611084
# Modify:   
   r1 r2, r3: 0.001, 0.001, 0.001 
   Train_Loss:0.6739174415325296   test_MSE:1.0329493284225464    test_mae: 0.7891864776611328
# Modify:   
   r1 r2, r3: 0.001, 0.001, 0.001 
   Train_Loss:0.718031425585692   test_MSE:1.0329493284225464    test_mae: 0.7891864776611328
# Modify:   
   r1 r2, r3: 0.001, 0.001, 0.001 
   Train_Loss:0.6550805854470763   test_MSE:1.0381864309310913    test_mae: 0.7875375747680664
# Modify:   
   r1 r2, r3: 0.001, 0.001, 0.001 
   Train_Loss:0.6550805854470763   test_MSE:1.0381864309310913    test_mae: 0.7875375747680664
# Modify:   
   r1 r2, r3: 0.001, 0.001, 0.001 
   Train_Loss:0.6550805854470763   test_MSE:1.0381864309310913    test_mae: 0.7875375747680664
# Modify:   
   r1 r2, r3: 0.001, 0.001, 0.001 
   Train_Loss:0.6398214391759924   test_MSE:1.035434603691101    test_mae: 0.7901739478111267
# Modify:   
   r1 r2, r3: 0.001, 0.001, 0.001 
   Train_Loss:0.6295214414596557   test_MSE:1.0386199951171875    test_mae: 0.7903924584388733
# Modify:   
   r1 r2, r3: 0.001, 0.001, 0.001 
   Train_Loss:0.718031425585692   test_MSE:1.0329493284225464    test_mae: 0.7891864776611328
# Modify:   
   r1 r2, r3: 0.001, 0.001, 0.001 
   Train_Loss:0.7205288288268176   test_MSE:1.0322469472885132    test_mae: 0.7889722585678101
# Modify:    embed_base 
   r1 r2, r3: 0.001, 0.001, 0.001 
   Train_Loss:0.7205288288268176   test_MSE:1.0322469472885132    test_mae: 0.7889722585678101
# Modify:   embed_gat
   r1 r2, r3: 0.001, 0.001, 0.001 
   Train_Loss:0.8039689660072327   test_MSE:1.052633285522461    test_mae: 0.8094300627708435
# Modify:   embed_egat  new_graph
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:0.6807798309759661   test_MSE:1.0363922119140625    test_mae: 0.7949941158294678
# Modify:   embed_egat  每20个epoch减少学习率
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:0.6902761066501791   test_MSE:1.0353935956954956    test_mae: 0.7939141392707825
# Modify:    embed_egat  graph
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:0.7949946549805728   test_MSE:1.0804065465927124    test_mae: 0.8343279361724854
# Modify:   embed_gat graph 可以看到正确的图效果反而更差了
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:0.7720466418699785   test_MSE:1.0781254768371582    test_mae: 0.826403796672821
# Modify:    embed_gat graph 只保留用户到商品的边
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:1.1789686544374987   test_MSE:1.1158994436264038    test_mae: 0.8673509955406189
# Modify:   embed_gat graph 只保留商品到用户的边
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:0.7349705052646723   test_MSE:1.0695377588272095    test_mae: 0.8241719007492065
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:0.735711511563171   test_MSE:1.0743496417999268    test_mae: 0.8291066884994507
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:0.7142177115787159   test_MSE:1.0403811931610107    test_mae: 0.7966673970222473
# Modify:   融合 3个 base gat egat
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:0.6809072690931234   test_MSE:1.031583547592163    test_mae: 0.7897380590438843
# Modify:   融合base egat
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:0.6520759605548598   test_MSE:1.0309090614318848    test_mae: 0.7891035079956055
# Modify:   new_graph ,终于比正确的图效果差了。。。。
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:0.6720038916576992   test_MSE:1.0314024686813354    test_mae: 0.7904279232025146
# Modify:   融合base gat
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:0.7005832086909901   test_MSE:1.032132863998413    test_mae: 0.7890003323554993
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:0.7260278090834618   test_MSE:1.033682942390442    test_mae: 0.7941799759864807
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:0.7180534167723223   test_MSE:1.0359504222869873    test_mae: 0.7944778800010681
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:0.7083954343741591   test_MSE:1.0329725742340088    test_mae: 0.7918416261672974
# Modify:   only trust user
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:0.6805756621890597   test_MSE:1.030436635017395    test_mae: 0.7907031774520874
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:0.6620131797260709   test_MSE:1.0305559635162354    test_mae: 0.791054904460907
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:0.6314184168974558   test_MSE:1.0317070484161377    test_mae: 0.7891152501106262
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:0.7083954343741591   test_MSE:1.0329725742340088    test_mae: 0.7918416261672974
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.01 
   Train_Loss:0.6901657750660722   test_MSE:1.030419111251831    test_mae: 0.7900748252868652
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.02 
   Train_Loss:0.6748665164817463   test_MSE:1.029778003692627    test_mae: 0.7900926470756531
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.02 
   Train_Loss:0.6748665164817463   test_MSE:1.029778003692627    test_mae: 0.7900926470756531
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.02 
   Train_Loss:0.523686408996582   test_MSE:0.9623932838439941    test_mae: 0.7299813628196716
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.02 
   Train_Loss:0.6759808158332651   test_MSE:1.030228853225708    test_mae: 0.7898157238960266
# Modify:   aug_r 效果并不好。。。
   r1 r2, r3: 0.001, 0.01, 0.02 
   Train_Loss:0.6732140251181342   test_MSE:1.0338973999023438    test_mae: 0.7957436442375183
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.02 
   Train_Loss:0.7037698186256669   test_MSE:1.0299861431121826    test_mae: 0.7895515561103821
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.02 
   Train_Loss:0.6848205835981802   test_MSE:1.0304975509643555    test_mae: 0.7905141115188599
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.02 
   Train_Loss:0.7131613154302944   test_MSE:1.0306085348129272    test_mae: 0.7912053465843201
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.02 
   Train_Loss:0.6947164589708502   test_MSE:1.030411720275879    test_mae: 0.7895740866661072
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.02 
   Train_Loss:0.7037698186256669   test_MSE:1.0299861431121826    test_mae: 0.7895515561103821
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.01 
   Train_Loss:0.6728829761797731   test_MSE:1.0312678813934326    test_mae: 0.791279673576355
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.02 
   Train_Loss:0.7144449746066873   test_MSE:1.0303717851638794    test_mae: 0.789808452129364
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.02 
   Train_Loss:0.6944128281690858   test_MSE:1.0301470756530762    test_mae: 0.7899881601333618
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.02 
   Train_Loss:0.7252448851411993   test_MSE:1.031999111175537    test_mae: 0.7935479283332825
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.02 
   Train_Loss:0.6904903955080293   test_MSE:1.0315444469451904    test_mae: 0.79231858253479
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.02 
   Train_Loss:0.734416055408391   test_MSE:1.0317314863204956    test_mae: 0.7891663908958435
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.02 
   Train_Loss:0.6755685894326731   test_MSE:1.030870795249939    test_mae: 0.7884732484817505
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.02 
   Train_Loss:0.6944128281690858   test_MSE:1.0301470756530762    test_mae: 0.7899881601333618
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.02 
   Train_Loss:0.6923223544250835   test_MSE:1.030558705329895    test_mae: 0.7902041077613831
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.02 
   Train_Loss:0.6937314549630339   test_MSE:1.0301629304885864    test_mae: 0.7902209758758545
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.02 
   Train_Loss:0.6843829127875242   test_MSE:1.029866337776184    test_mae: 0.789316713809967
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.02 
   Train_Loss:0.6891374283216216   test_MSE:1.0304436683654785    test_mae: 0.7897456884384155
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.02 
   Train_Loss:0.6809540851549669   test_MSE:1.0370162725448608    test_mae: 0.7932710647583008
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.02 
   Train_Loss:0.674855999648571   test_MSE:1.0405385494232178    test_mae: 0.7926000952720642
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.02 
   Train_Loss:0.7313782904635776   test_MSE:1.0369453430175781    test_mae: 0.7976240515708923
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.01 
   Train_Loss:0.7975833551450209   test_MSE:1.02975332736969    test_mae: 0.7918297052383423
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:0.6791634573177858   test_MSE:1.0303438901901245    test_mae: 0.7899335622787476
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.01 
   Train_Loss:0.7975833551450209   test_MSE:1.02975332736969    test_mae: 0.7918297052383423
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.01 
   Train_Loss:0.5810839076836903   test_MSE:0.9555263519287109    test_mae: 0.721472978591919
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.01 
   Train_Loss:0.560549521446228   test_MSE:0.9519623517990112    test_mae: 0.7214834690093994
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:1.128690635616129   test_MSE:1.105853796005249    test_mae: 0.8424587845802307
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:1.102617851712487   test_MSE:1.0967369079589844    test_mae: 0.8699366450309753
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:1.0431218418208035   test_MSE:1.0662063360214233    test_mae: 0.8239110112190247
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.001 
   Train_Loss:1.0479882007295436   test_MSE:1.076421856880188    test_mae: 0.8380221128463745
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.0001 
   Train_Loss:1.0810972533442758   test_MSE:1.0852690935134888    test_mae: 0.8383777141571045
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.04 
   Train_Loss:0.8622890927574851   test_MSE:1.0344467163085938    test_mae: 0.7963628172874451
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.04 
   Train_Loss:0.7829856994477186   test_MSE:1.0301586389541626    test_mae: 0.7886434197425842
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.05 
   Train_Loss:0.7990748543630947   test_MSE:1.030084252357483    test_mae: 0.7896595597267151
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.06 
   Train_Loss:0.8799923942847685   test_MSE:1.0335729122161865    test_mae: 0.7961770296096802
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.05 
   Train_Loss:0.7608649256554517   test_MSE:1.0300549268722534    test_mae: 0.7904466390609741
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.7974889115853743   test_MSE:1.0350617170333862    test_mae: 0.7965861558914185
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.6962473798881877   test_MSE:1.0314695835113525    test_mae: 0.7896028757095337
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.7458159449425611   test_MSE:1.0301350355148315    test_mae: 0.7887704372406006
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.7252256409688429   test_MSE:1.0298866033554077    test_mae: 0.7906440496444702
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.7252256409688429   test_MSE:1.0295718908309937    test_mae: 0.7889062762260437
 # Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.7252256409688429   test_MSE:1.0295718908309937    test_mae: 0.7889062762260437
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.7463222972371362   test_MSE:1.030482530593872    test_mae: 0.7898196578025818
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.7502594529227777   test_MSE:1.0302059650421143    test_mae: 0.7899062037467957
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.8295416127551686   test_MSE:1.0330142974853516    test_mae: 0.796913743019104
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.7270664484663443   test_MSE:1.0300283432006836    test_mae: 0.7901684641838074
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.7458159449425611   test_MSE:1.0301350355148315    test_mae: 0.7887704372406006
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.7829856994477186   test_MSE:1.0301586389541626    test_mae: 0.7886434197425842
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.7798364379189231   test_MSE:1.0334713459014893    test_mae: 0.7973083257675171
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.9105006822130897   test_MSE:1.04034423828125    test_mae: 0.8042464852333069
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.7458159449425611   test_MSE:1.0301350355148315    test_mae: 0.7887704372406006
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.7362945662303404   test_MSE:1.030900478363037    test_mae: 0.7893516421318054
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.6716339073397897   test_MSE:1.0318286418914795    test_mae: 0.7879211902618408
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.7367423014207319   test_MSE:1.0309921503067017    test_mae: 0.7908459305763245
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.7367423014207319   test_MSE:1.0309921503067017    test_mae: 0.7908459305763245
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.753466172651811   test_MSE:1.030457854270935    test_mae: 0.79122394323349
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.7252256409688429   test_MSE:1.0298866033554077    test_mae: 0.7906440496444702
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.824432910843329   test_MSE:1.0309652090072632    test_mae: 0.7930421829223633
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.7680751803246412   test_MSE:1.0300774574279785    test_mae: 0.7924071550369263
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.7187142331491817   test_MSE:1.0316685438156128    test_mae: 0.7890979647636414
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.7871679799123243   test_MSE:1.0343594551086426    test_mae: 0.7894496321678162
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.7116776176474311   test_MSE:1.031708002090454    test_mae: 0.792158305644989
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.752737507224083   test_MSE:1.0300575494766235    test_mae: 0.790332019329071
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.7037888636643236   test_MSE:1.02996826171875    test_mae: 0.7899472117424011
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.7194680659608408   test_MSE:1.0298839807510376    test_mae: 0.7887093424797058
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.8576188468270831   test_MSE:1.0308527946472168    test_mae: 0.7970637083053589
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.8576188468270831   test_MSE:1.0309323072433472    test_mae: 0.7975044250488281
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.7592507849136988   test_MSE:1.0289685726165771    test_mae: 0.7929573059082031
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.9363956534200244   test_MSE:1.0412003993988037    test_mae: 0.8108334541320801
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.7194680659608408   test_MSE:1.0295201539993286    test_mae: 0.7872069478034973
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.7194680659608408   test_MSE:1.0295201539993286    test_mae: 0.7872069478034973
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:2.810793789950284   test_MSE:1.0296037197113037    test_mae: 0.7891899943351746
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:1.0927512618628414   test_MSE:1.0295017957687378    test_mae: 0.787727952003479
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:1.034232410517606   test_MSE:1.0296169519424438    test_mae: 0.787794291973114
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:3.2453694939613342   test_MSE:1.0302151441574097    test_mae: 0.7863870859146118
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:1.392570129849694   test_MSE:1.0297931432724    test_mae: 0.7875320315361023
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.1 
   Train_Loss:0.6123810816894878   test_MSE:1.083945393562317    test_mae: 0.811475396156311
# Modify:   
   r1 r2, r3: -0.0001, 0.01, 0.01 
   Train_Loss:0.7001291621815074   test_MSE:1.0408700704574585    test_mae: 0.7933495044708252
# Modify:   
   r1 r2, r3: 0.001, 0.01, 0.01 
   Train_Loss:0.7163406258279627   test_MSE:1.0423829555511475    test_mae: 0.7956116795539856


(GDS) [lgq 21:52:51 myMCL]$ python3 main1.py 
Data info: user_num: 7375, item_num 105114, rating record num: 282650 
trust_num: 111781
Get statistic info:   k1= 1
done
epoch : 1       train Loss:     test RMSE: 0.9653780823908272    test MAE:  0.725658788868101
(GDS) [lgq 21:53:09 myMCL]$ python3 main1.py 
Data info: user_num: 7375, item_num 105114, rating record num: 282650 
trust_num: 111781
Get statistic info:   k1= 2
done
epoch : 1       train Loss:     test RMSE: 0.957663904894232    test MAE:  0.7227939227315247
(GDS) [lgq 21:53:30 myMCL]$ 
(GDS) [lgq 21:53:32 myMCL]$ python3 main1.py 
Data info: user_num: 7375, item_num 105114, rating record num: 282650 
trust_num: 111781
Get statistic info:   k1= 3
done
epoch : 1       train Loss:     test RMSE: 0.9566105988698961    test MAE:  0.7231158562428529
(GDS) [lgq 21:53:38 myMCL]$ python3 main1.py 
Data info: user_num: 7375, item_num 105114, rating record num: 282650 
trust_num: 111781
Get statistic info:   k1= 4
done
epoch : 1       train Loss:     test RMSE: 0.9574354715148363    test MAE:  0.7242769364215788
(GDS) [lgq 21:53:50 myMCL]$ python3 main1.py 
Data info: user_num: 7375, item_num 105114, rating record num: 282650 
trust_num: 111781
Get statistic info:   k1= 5
done
epoch : 1       train Loss:     test RMSE: 0.9588940482795579    test MAE:  0.7257206155076344
(GDS) [lgq 21:53:59 myMCL]$ python3 main1.py 
Data info: user_num: 7375, item_num 105114, rating record num: 282650 
trust_num: 111781
Get statistic info:   k1= 6
done
epoch : 1       train Loss:     test RMSE: 0.9605615885852673    test MAE:  0.7272442819545608
(GDS) [lgq 21:54:09 myMCL]$ python3 main1.py 
Data info: user_num: 7375, item_num 105114, rating record num: 282650 
trust_num: 111781
Get statistic info:   k1= 7
done
epoch : 1       train Loss:     test RMSE: 0.962260929441682    test MAE:  0.728719135756661
(GDS) [lgq 21:54:15 myMCL]$ python3 main1.py 
Data info: user_num: 7375, item_num 105114, rating record num: 282650 
trust_num: 111781
Get statistic info:   k1= 8
done
epoch : 1       train Loss:     test RMSE: 0.9639292030359948    test MAE:  0.7301215290880738
(GDS) [lgq 21:54:21 myMCL]$ python3 main1.py 
Data info: user_num: 7375, item_num 105114, rating record num: 282650 
trust_num: 111781
Get statistic info:   k1= 9
done
epoch : 1       train Loss:     test RMSE: 0.9655374977210552    test MAE:  0.7314316801403192
(GDS) [lgq 21:54:26 myMCL]$ python3 main1.py 
Data info: user_num: 7375, item_num 105114, rating record num: 282650 
trust_num: 111781
Get statistic info:   k1= 10
done
epoch : 1       train Loss:     test RMSE: 0.9670710065518685    test MAE:  0.7326670007815086
(GDS) [lgq 21:54:36 myMCL]$ 