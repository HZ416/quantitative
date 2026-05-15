# A股周频量化交易的SVM技术指标研究报告

本报告聚焦A股“每周调仓一次”的低频量化，用SVM对技术指标建模：给出≥20个候选指标、周频特征工程与时间序列验证/回测流程，并结合交易规则与成本约束提供10组优先实验组合与可复现代码框架。citeturn19search5turn1search3turn20view0

## 研究目标与关键假设

### 研究目标与核心假设
研究目标：在A股场景下，构建**周频（weekly）可交易**、且对**SVM（线性/核）友好**的技术指标特征集，并建立稳健的筛选、训练与样本外验证流程，尽量降低数据泄漏与过拟合风险。citeturn19search5turn21search1turn1search3

核心假设（可作为默认设定，未指定样本期则不限制）：
- **周频交易定义**：以“周”为最小决策单元（典型做法：周五收盘后生成信号，下一交易周开盘执行；或用“每周最后一个交易日收盘价”作为周线收盘）。此设置旨在遵守时间顺序、避免用到未来信息。citeturn1search3turn18search3  
- **预测目标**（二选一，更推荐先做分类）：  
  - 方向分类：预测下周收益符号 \(y=\mathbb{1}(r_{t\to t+1w}>0)\)。  
  - 分位分类：做Top/Bottom分位（中间分位“空缺/不交易”）以降低噪声与换手。citeturn21search1turn17search2  
- **回测频率**：每周再平衡一次（周频下单），但PnL可以按日/按周记账；风险控制可按日监控（例如遇到涨跌停/停牌、无法成交）。citeturn5view0turn6view2  

### A股机制与约束（影响“周频指标可交易性”）
A股交易制度会直接影响周频信号落地与回测模拟：
- 多数股票存在**日涨跌幅限制（常见±10%）**；风险警示股票与退市整理期股票的限制不同（如风险警示常见±5%、退市整理常见±10%等），且不同板块（如科创板/创业板）规则更高或有例外条款。citeturn5view0turn6view1turn6view2turn1search9  
- 创业板注册制股票：上市后前5个交易日不设涨跌幅限制，之后竞价交易涨跌幅限制比例为20%。citeturn1search9  
- 交易所规则允许在监管批准下调整涨跌幅限制。citeturn5view0turn6view2  
- 规则存在**变更不确定性**：例如2025年深交所曾发布“主板风险警示股票涨跌幅拟调整为10%”的征求意见稿，意味着研究与回测应把“涨跌幅/规则参数”做成可配置项。citeturn6view3  

### 交易成本与回测成本假设（未指定时给常见选项）
A股回测常见成本由：佣金、印花税、过户费、交易所经手费（以及滑点/冲击成本）构成。监管与机构公告对其中部分费率给出明确规定：  
- 证券交易印花税：自2023-08-28起“减半征收”（市场通常按“卖出收取、单边征收”模拟）。citeturn2view0turn1search12  
- 佣金：A股、基金等交易佣金实行“最高上限向下浮动”，上限不高于成交金额的3‰，并常见“最低5元”规则（适用于很多券商口径）。citeturn10search4  
- 过户费：中国结算宣布自2022-04-29起股票交易过户费总体下调50%，沪深A股统一为按成交金额0.01‰双向收取。citeturn9search10  
- 股票交易经手费：上交所公告自2023-08-28起将A/B股经手费由0.00487%双向下调至0.00341%双向。citeturn9search4  

下表给出**周频回测**常用的“全成本”配置方式（建议做敏感性分析，不要只跑单一成本）：citeturn2view0turn10search4turn9search10turn9search4  

| 成本项 | 权威/公开口径 | 周频回测建议写法 | 备注 |
|---|---|---|---|
| 佣金 | 上限≤成交额3‰、向下浮动citeturn10search4 | 设为可协商区间（如万1~万3）双向 | 周频换手较低但仍敏感 |
| 印花税 | 2023-08-28起减半征收citeturn2view0turn1search12 | 卖出单边计提（如0.05%口径常用于模拟） | 费率需随政策更新 |
| 过户费 | 0.01‰双向citeturn9search10 | 双向计提 | 对高换手策略更敏感 |
| 经手费 | 0.00341%双向（上交所口径）citeturn9search4 | 双向计提（沪深可统一处理） | 建议写成参数 |
| 滑点/冲击 | 无统一固定费率 | 用bps区间（如单边1~10bps）或成交概率模型 | 周频也会遇到涨跌停/流动性不足 |

## 技术指标候选集与适配性分析

### 选择“周频 + SVM”指标的原则
周频技术指标与日频相比，一般会带来两类结构性变化：  
- **噪声—样本量权衡**：周频聚合往往降低高频噪声，但样本点更少（单标的每年约50个样本），更依赖跨标的合并训练或更长样本期。周频方向预测与对照基准在文献中被专门讨论。citeturn17search2turn17search13  
- **SVM对特征表示的敏感性**：SVM（尤其RBF核）对特征尺度与分布非常敏感，必须把“缩放/缺失值/数据泄漏防控”纳入特征设计，而不是事后补丁。citeturn19search5turn22search1turn21search1  

下面给出≥20个指标候选集。**指标公式与常用参数**主要参考entity["company","同花顺","fintech firm, china"]量化数据平台“技术因子”与entity["company","米筐","quant platform, china"]文档“技术分析”。citeturn20view0turn12view0  

### 指标候选集总表（不少于20个）
表中“周频适配”强调：是否能在周线OHLCV上稳定计算、是否过度依赖日内结构；“SVM要点”强调：尺度、分布、缺失值与非线性表达。citeturn20view0turn12view0turn22search1turn21search1  

| 指标 | 类别 | 周频适配性（理由） | 对SVM的适配要点 | 周频参数建议（经验起点） |
|---|---|---|---|---|
| MA（简单均线）citeturn20view0 | 趋势/平滑 | 高：周线更平滑、减少假信号 | 用“价/均线比”“均线斜率”“交叉事件”比直接价格更稳；需缩放 | 4、8、12、26周 |
| EXPMA/EMA（指数均线）citeturn20view0turn12view0 | 趋势/平滑 | 高：对趋势转折更敏感 | 与MA同；注意EMA递推对缺失周要先补齐周序列 | 4、8、12、26周 |
| BBI（多空指数）citeturn20view0 | 趋势 | 高：多周期均线平均，周频更稳 | 尺度接近价格，建议转为比例（BBI/Close）并标准化 | (3,6,12,24)周或(2,4,8,16)周 |
| DMA（均线差）citeturn20view0 | 趋势 | 高：差分能去除部分非平稳 | 差分型特征更接近平稳；对线性SVM友好 | 10-50周差、再平滑10周 |
| MACDciteturn20view0turn12view0 | 趋势/动量 | 高：在周线常用于捕捉中期趋势 | 用DIFF/DEA/柱体及其变化率；需标准化 | (12,26,9)“周参数”或折算为(3,6,3)周起步 |
| TRIXciteturn20view0turn12view0 | 趋势/动量 | 中高：三次EMA更平滑，周频信号少但稳 | 输出为百分比变化，尺度较友好；仍建议标准化 | 12周TRIX + 20周均线 |
| PRICEOSC（价格振荡）citeturn20view0 | 趋势/摆动 | 中：本质是两条MA相对差 | 比例型特征对尺度更稳；可与波动率交互 | (12,26)周 |
| DMI/ADX（DMI含ADX）citeturn12view0turn20view0 | 趋势强度/波动 | 高：适合周线衡量趋势强弱 | ADX天然0-100附近，尺度好；用“趋势强度过滤器”很实用 | 14周（或10周） |
| DDIciteturn20view0 | 趋势/方向偏离 | 中：公式较复杂，但周线可算 | 分母/比例结构对尺度友好；关注缺失周导致不连续 | 13周/30周等默认参数起步 |
| MTM（动量）citeturn20view0 | 动量 | 高：周度动量与周频持有期一致 | 用多滞后动量（1、4、12周）+标准化；对线性SVM常有效 | N=4、12、26周 |
| ROC（变动速率）citeturn20view0 | 动量 | 高：百分比形式更稳 | 比值型减少价格量纲；仍需对异常点winsorize | N=4、12、26周 |
| RSIciteturn20view0turn12view0 | 摆动/超买超卖 | 高：0-100有界，周频更抗噪 | 有界特征对RBF核友好；可做分箱（如<30、30-70、>70） | 6、12、24周三组 |
| KDJciteturn20view0turn12view0 | 摆动 | 中高：周线KDJ信号更少但更“慢” | 有界特征；用K、D、J及其差（K-D） | (9,3,3)周起步，或(12,3,3)周 |
| WR（威廉指标）citeturn20view0turn12view0 | 摆动 | 中高：同为有界震荡 | 与RSI高度相关，需做冗余控制 | 10周/6周双版本 |
| CCIciteturn20view0 | 顺势/偏离 | 中：无界，易受极端波动影响 | 强烈建议标准化或rank化；可加截尾 | 14周或20周 |
| BIAS（乖离率）citeturn20view0turn12view0 | 反趋向/均值回归 | 高：周频下更像“偏离估值” | 天然百分比；可与波动率交互控制假信号 | 12周、26周 |
| DPOciteturn20view0turn12view0 | 周期/摆动 | 中：强调周期性，周线更适用长周期 | 输出在价格量纲，建议转为比例或z-score | 20周，位移=6周 |
| DBCDciteturn20view0 | 反趋向/加速 | 中：偏“二阶差分”思想 | 差分+平滑对线性SVM友好；对噪声敏感需周频平滑 | 用BIAS差分+SMA平滑 |
| SI（摆动指标）citeturn20view0 | 摆动 | 中：对开高低收更敏感 | 不同市场开盘跳空会影响；需标准化并检查缺失 | 用默认实现，配合稳健缩放 |
| ARBRciteturn12view0 | 情绪/强弱 | 中：依赖H/O/L组合，周线可用 | 数值可能偏离稳定区间，建议rank化 | 26周 |
| VR（量比）citeturn20view0turn12view0 | 成交量 | 低到中：原定义偏日内“每分钟”，周频需改写 | 体量跨度极大，必须取log或rank；缺失量需处理 | 周频可用“本周均量/过去k周均量”替代 |
| VMA（成交量均线）citeturn20view0 | 成交量 | 中高：周成交量做均线更稳定 | 使用log(Volume)后再做均线/差分更稳 | 4、12、26周 |
| VOSC（成交量振荡）citeturn20view0 | 成交量 | 中：量MA相对差，周频可用 | 建议rank化；与价格动量联立更有意义 | (12,26)周 |
| VROC（量变动速率）citeturn20view0 | 成交量 | 中：周成交量波动大、偏态强 | 需log差分或winsorize；避免被“事件周”支配 | 12周 |
| VSTD（量标准差）citeturn20view0 | 成交量/波动 | 中：周数据样本少时估计不稳 | 建议更长窗口+稳健缩放 | 26周 |
| MFI（资金流量指标）citeturn20view0 | 量价/强弱 | 中高：结合价格与量，周频有意义 | 0-100有界；与RSI相关但不完全重合 | 14周 |
| OBVciteturn20view0 | 量价 | 中：累计量值会爆炸 | 强烈建议使用“OBV变化率/差分/滚动z-score”而非原值 | 计算后取ΔOBV(1,4,12周) |
| PVTciteturn20view0 | 量价趋势 | 中：累计量价，周频可用 | 同OBV：差分、标准化、rank化 | 用ΔPVT替代PVT |
| WVADciteturn20view0 | 量价/离散 | 中：对“收盘-开盘”敏感 | 对跳空与事件敏感；用稳健缩放 | (24,6)周起步 |
| ATRciteturn20view0 | 波动率 | 高：适合周频做仓位/止损/风险预算 | 用ATR/Close（归一化）提升跨股可比性 | 14周 |
| STD（收益/价格标准差）citeturn20view0turn12view0 | 波动率 | 高：周收益STD可直接做风险因子 | 标准化时要在训练折内拟合，防泄漏 | 26周或52周 |
| MASS（梅丝线）citeturn20view0 | 波动结构 | 中：对高低价区间的结构变化敏感 | 指标相对小众，需先做稳定性检验再入模 | (9,25)周 |

**哪些指标常见但对“周频 + SVM”不友好？（建议谨慎或改造）**  
- “强事件驱动/强日内定义”的成交量指标（如原始口径的VR）在周频上往往需要**重新定义**，否则含义漂移。citeturn20view0  
- “累计型”指标（OBV、PVT等）直接入模会被股本规模、历史长度、复权处理影响，建议用差分/增长率/滚动z-score替代。citeturn20view0turn21search1  
- 彼此高度同源的指标（例如MA族、MACD/PRICEOSC、RSI/WR/MFI）容易造成冗余与多重检验偏差，需要用相关性/聚类/嵌入式选择做降维与稳健性验证。citeturn21search1turn15search19turn18search1  

## 特征工程与SVM建模建议

### 周频数据构造与“可交易”对齐
推荐从日线构造周线OHLCV（避免直接用供应商“周线”而不清楚聚合方式），并严格做到：
- 特征只使用截至周 \(t\) 收盘的数据；
- 标签使用未来一周 \(t\to t+1\) 的收益（或分位类别）；
- 交易执行在 \(t+1\) 周开盘或下一交易日开盘，以避免“收盘即成交”的不现实假设。citeturn21search1turn1search3  

### 面向周频的特征形态清单
以下是对SVM更友好的周频特征工程形态（“原始值 + 变化 + 相对化”三件套）：

**价格与指标的原始值（Level）**  
- RSI、MFI、ADX这类有界/半有界指标可直接使用（再缩放）；  
- MA、BBI、BOLL中轨这类“价格量纲”的指标，建议转为：\(\text{Close}/\text{MA}-1\)、\(\text{BOLL\_width}=(Upper-Lower)/MID\)。citeturn20view0turn12view0turn22search0  

**差分/涨跌（Delta）**  
- \(\Delta\)RSI、\(\Delta\)MACD柱、\(\Delta\)ADX能更直接表达“加速/减速”，也往往更接近平稳，对线性SVM更友好。citeturn20view0turn12view0turn19search5  

**滞后项（Lag）**  
- 周频常用1~4周滞后（短期记忆）与12~26周滞后（中期结构）。  
- 把“本周指标 + 上周指标 + 变化率”并列，有助于SVM学习非线性阈值（例如RSI从45跃升到60）。citeturn1search6turn22search16  

**滚动统计（Rolling stats）**  
- 对每个指标做滚动均值/标准差/分位数（如26周窗口），形成“相对位置”特征：  
  \[
  z_t=\frac{x_t-\mu_{t,26}}{\sigma_{t,26}}
  \]
  这会显著增强跨股票、跨时期可比性。citeturn22search0turn21search1  

**归一化与稳健缩放（Scaling）**  
- SVM对特征尺度敏感，建议默认在Pipeline里加入StandardScaler（或对强偏态用RobustScaler）。citeturn22search1turn19search2turn21search1  

**分箱（Binning）与非线性门槛**  
- 将RSI/MFI/WR等有界指标分箱（如<30、30-70、>70）能把经验阈值变成可学习的离散结构；对线性SVM尤其有用。citeturn20view0turn19search5  

**主成分（PCA）与去冗余**  
- 面对多指标强相关，PCA可作为“降维基线”，但必须在训练折内拟合以防泄漏。citeturn21search1turn17search6  

**交互项（Interactions）**
- 周频常见有效交互：  
  - 趋势×波动：\((\text{Close}/MA-1) \times ATR/Close\)（趋势强但波动扩张时的行为可能不同）；  
  - 动量×量能：ROC × VROC（“放量上涨”与“缩量上涨”在A股差异明显）。citeturn20view0turn17search6  

### SVM（线性/核）适配策略与超参数建议
scikit-learn文档对SVM优势与关键参数有明确定义：RBF核主要关注C与gamma；SVC训练开销随样本量至少二次增长，样本很大时需考虑线性模型或核近似。citeturn1search6turn22search3  

**线性SVM（baseline优先）**  
- 适用：指标数很多、样本量也大（跨股合并）；或希望更可解释。citeturn22search3turn19search5  
- 建议：  
  - 先用线性SVM做基线（例如C取对数网格：0.01、0.1、1、10）。  
  - 配合L1/L2正则（若用LinearSVC等）做稀疏化倾向的特征筛选。citeturn22search3turn21search2  

**核SVM（RBF核为主）**  
- 适用：你相信“指标—收益方向”关系存在明显非线性阈值/形态（在技术指标场景很常见）。citeturn1search6turn22search16  
- 关键超参数：  
  - C：控制误分类惩罚与边界平滑程度；  
  - gamma：控制单一样本影响范围（越大越“局部”）。citeturn1search6turn22search16  
- 默认建议：  
  - 使用gamma='scale'起步（scikit-learn默认逻辑），再做网格/随机搜索；  
  - C与gamma用对数空间搜索。例如：C∈[0.1,1,10,100]，gamma∈[1e-3,1e-2,1e-1,1]（需结合特征维度与缩放后方差调整）。citeturn22search3turn21search2turn22search16  

**类别不平衡与class_weight**  
A股“下周涨/跌”标签通常不会严格50/50，且你可能引入“中性不交易”导致更强不平衡。SVC支持class_weight（如'balanced'）来缓解。citeturn22search3turn0search3  

**缺失值处理（停牌/数据空洞/指标warm-up）**  
- 指标需要回溯窗口，前若干周必然产生NaN；停牌与无成交也会带来缺失。  
- 用SimpleImputer等做缺失值填补，但必须置于Pipeline并在训练折内fit，避免泄漏。citeturn21search0turn21search1  

**概率输出（用于仓位/风控）**  
若需要概率而非硬分类，可用CalibratedClassifierCV做校准；文档指出它与SVC(probability=True)的关系与实现细节。citeturn21search3turn21search7  

## 统计检验、验证与回测框架

### 用于“筛选与验证指标”的统计方法组合
技术指标研究最常见的陷阱是“指标太多 + 试太多组合”，容易产生数据挖掘偏误（data snooping）。White提出Reality Check方法应对数据重复使用导致的虚假显著性；Hansen提出SPA检验提高功效并降低对劣质候选的敏感性。citeturn15search19turn18search1  

一个更稳健的“由浅入深”验证栈建议如下：

**相关性与冗余控制**  
- 先做特征间相关性与聚类（尤其MA族/动量族），删除强共线特征，减少“多重比较”。citeturn21search1turn15search19  

**信息系数IC（Information Coefficient）与稳定性**  
- IC常用于衡量“信号对未来收益的解释力”，可用Pearson（线性）或Spearman（秩相关）计算。citeturn15search17turn15search5  
- 周频设置：每周横截面计算IC（用信号值 vs 下周收益），再观察IC时间序列的均值、波动与分位（例如滚动一年）。citeturn15search17turn17search2  

**时间序列交叉验证（Time Series CV）**  
- scikit-learn的TimeSeriesSplit为时间有序数据提供训练/测试划分，避免“用未来训练、用过去测试”的错误。citeturn1search3  
- 对周频交易更贴合的做法是“滚动窗口/扩展窗口”的walk-forward评估（在预测与交易领域被系统化讨论）。citeturn18search3turn18search7  

下面给出一个简化的“周频TimeSeriesSplit”示意（每折测试集为连续的未来区间）：citeturn1search3turn18search3  

```mermaid
flowchart TD
  A[按周排序的样本 1..T] --> B[Fold 1: Train=1..t1, Test=t1+1..t2]
  A --> C[Fold 2: Train=1..t2, Test=t2+1..t3]
  A --> D[Fold k: Train=1..t(k), Test=t(k)+1..t(k+1)]
  B --> E[记录每折: AUC/Accuracy + 策略收益指标]
  C --> E
  D --> E
```

**样本外回测（OOS backtest）与嵌套调参**  
- GridSearchCV用于参数搜索，但必须把“缩放/填补/降维”等预处理放入Pipeline，并在训练集内做交叉验证，评估集保持完全隔离。citeturn21search2turn21search1  
- 若你同时做“特征选择 + 调参 + 策略规则搜索”，建议引入更严格的抗过拟合度量：  
  - Deflated Sharpe Ratio（修正选择偏差/非正态/多次试验影响）；  
  - Probability of Backtest Overfitting（PBO）。citeturn16search0turn16search1  

**模型预测准确性比较（可选）**  
若需要在统计意义上比较两种模型预测误差，可用Diebold–Mariano检验作为预测精度比较的经典方法之一。citeturn15search2turn15search6  

### 性能度量（交易视角）与过拟合防控清单
交易绩效建议同时报告**收益、风险、稳定性**三类指标，并把“回测过拟合概率”类指标作为补充：  
- 年化收益、年化波动、夏普、最大回撤、胜率、收益回撤比、信息比率（若对基准做超额）。  
- 抗过拟合：Reality Check / SPA控制数据挖掘偏误，DSR与PBO控制“试得越多看起来越好”的幻觉。citeturn15search19turn18search1turn16search0turn16search1  

技术指标有效性的大样本检验在经典研究中已有系统性讨论（例如对移动平均、区间突破等规则使用长样本与bootstrap）。这提醒我们：技术指标并非必然无效，但必须使用稳健的统计检验与样本外框架。citeturn16search3turn16search7  

## 实证流程与优先实验组合

### 可复现的实证流程总览
参考scikit-learn对“避免数据泄漏”的实践建议，流程应先分割训练/测试，再在训练折内拟合预处理与模型。citeturn21search1turn21search2  

```mermaid
flowchart LR
  A[数据获取: 日线OHLCV/复权/停牌标记] --> B[周线聚合: 周OHLCV]
  B --> C[特征构建: 指标+衍生(差分/滞后/z-score)]
  C --> D[训练集内预处理: 缺失值->缩放->降维(可选)]
  D --> E[时间序列CV: TimeSeriesSplit/Walk-forward]
  E --> F[超参数调优: C/gamma/class_weight]
  F --> G[样本外回测: 成交约束+成本+风控]
  G --> H[稳健性: 多期/多行业/子样本/RC或SPA/DSR/PBO]
```

### 分步输入/输出与关键参数表
（你可以把这张表当做“实验配置清单”，便于复现实验与做消融）citeturn20view0turn21search1turn1search3turn2view0turn9search10turn9search4  

| 步骤 | 输入 | 输出 | 关键参数/注意点 |
|---|---|---|---|
| 数据获取 | 日线OHLCV、成交额/量、停牌/退市/风险警示标记 | 原始面板数据 | 确认复权口径；保留“是否可交易”字段 |
| 周线聚合 | 日线序列 | 周OHLCV | 周定义（周五/最后交易日）；缺失周补齐 |
| 指标计算 | 周OHLCV | 指标Level | 指标窗口按“周”设定；warm-up NaN处理 |
| 特征衍生 | 指标Level | Δ、Lag、z-score、分箱、交互项 | 训练折内滚动统计，避免泄漏citeturn21search1 |
| 特征选择 | 全特征 | 子集/因子组合 | 相关性过滤+IC稳定性+嵌入式选择citeturn15search19turn15search17 |
| 模型训练 | 训练集特征与标签 | SVM模型 | 缩放必做citeturn22search1turn19search2；核/线性选择citeturn22search3turn1search6 |
| 调参验证 | 训练集 | 最优超参 | TimeSeriesSplitciteturn1search3；GridSearchCVciteturn21search2 |
| 回测执行 | 预测信号 | 交易序列/PnL | 成本：印花税/佣金/过户/经手费citeturn2view0turn10search4turn9search10turn9search4；涨跌停与无法成交citeturn6view1turn6view2 |
| 稳健性评估 | 回测结果 | 结论与置信度 | Reality Check/SPAciteturn15search19turn18search1；DSR/PBOciteturn16search0turn16search1 |

### 优先测试的10个“指标组合”（周频 + SVM友好）与理由
组合设计遵循两条经验原则：  
- 用“趋势 + 动量 + 波动/量能确认”构造相对正交的信号；  
- 优先选**有界/比例型/差分型**特征，减少尺度与极端值对SVM的伤害。citeturn22search1turn20view0turn17search6  

下表每组给出“最小可用组合”（后续可对每个指标做多窗口扩展）：

| 优先级组合 | 指标（建议用周频窗口） | 适用逻辑 | 为什么对SVM更友好 |
|---|---|---|---|
| 趋势基线组 | Close/MA、MACD柱、ADX | 趋势方向 + 趋势强度过滤 | 比例/有界/平滑，减少噪声citeturn20view0turn12view0 |
| 动量确认组 | MTM(12w)、ROC(12w)、PRICEOSC | 中期动量与均线差 | 多为差分/比率结构，易标准化citeturn20view0 |
| 均值回归组 | BIAS、RSI、CCI | 超买超卖 + 偏离修正 | RSI有界，BIAS为百分比，CCI需缩放citeturn20view0turn12view0 |
| 震荡择时组 | KDJ、WR、ΔRSI | 区间震荡内的拐点 | 多为0-100类指标，适合核SVM阈值学习citeturn20view0turn12view0turn1search6 |
| 波动控制组 | ATR/Close、STD(26w)、BOLL宽度 | 波动扩张/收敛与风险预算 | 归一化后跨股可比，利于SVM边界稳定citeturn20view0turn12view0 |
| 量价趋势组 | ΔOBV、ΔPVT、MFI | 量能是否支持价格方向 | 用差分避免累计爆炸；MFI有界citeturn20view0 |
| 放量突破组 | BOLL上轨突破事件、VROC、ROC | “突破 + 放量”筛选 | 事件特征+量能确认能显著非线性化citeturn20view0turn12view0 |
| 市场情绪组 | ARBR、VR(周化定义)、ADX | 情绪极端与趋势共振 | 需重定义VR；ARBR适合做状态变量citeturn12view0turn20view0 |
| 二阶变化组 | DBCD、ΔMACD、ΔADX | 捕捉加速/减速 | 差分增强平稳性；线性SVM也能学citeturn20view0turn1search6 |
| 小众结构探索组 | MASS、DDI、WVAD | 波动结构与离散量价 | 需用稳定性/子样本检验防“偶然有效”citeturn20view0turn15search19 |

### 默认实验设置建议（未指定时的常用值）
这些默认值强调“周频样本较少”这一现实，并兼顾SVC训练复杂度随样本增长至少二次的事实：citeturn22search3turn1search3  
- 样本长度：单标的至少5年（≈250周）；若跨股合并训练，建议10年以上覆盖多市场状态。citeturn17search2  
- 训练/测试切分：初始70/30或80/20；更推荐walk-forward（例如：训练3年→测试1年，滚动推进）。citeturn18search3turn1search3  
- 滚动统计窗口：26周（半年）与52周（一年）作为常用起点；短窗（8~12周）用于“快信号”。citeturn20view0turn12view0  
- 超参数搜索：先粗网格、再局部细化；避免在同一数据上无止境搜索（用RC/SPA/DSR/PBO约束研究过程）。citeturn15search19turn18search1turn16search0turn16search1  

## 风险、局限与A股特性

image_group{"layout":"carousel","aspect_ratio":"16:9","query":["Shanghai Stock Exchange building exterior","Shenzhen Stock Exchange building exterior","A-share stock market trading screen China"],"num_per_query":1}

### A股特性对“周频指标 + SVM”的影响
- **涨跌幅限制与无法成交**：当周频信号触发但标的处于涨停/跌停附近或被限制交易，实际无法按理论价格成交；回测若不模拟“成交失败/滑点扩大”，会高估策略表现。citeturn6view1turn6view2turn5view0  
- **风险警示与退市整理期差异**：风险警示股票与退市整理股票的涨跌幅限制比例不同；研究中是否剔除ST/退市整理会显著改变收益分布与可交易性。citeturn6view1turn6view2  
- **板块制度差异**：创业板涨跌幅规则与“上市前5日不设涨跌幅”等条款会改变极端收益与波动统计，从而影响指标阈值与SVM决策边界。citeturn1search9  
- **制度可能变化**：公开征求意见稿显示交易制度存在调整路径，建议将规则参数化并在不同制度情景下做稳健性回测。citeturn6view3turn4search3  

### 主要研究偏差与缓解措施
- **样本偏差（幸存者偏差/新股偏差）**：若只用当前成分股回测，会忽略退市与历史成分变动；周频策略尤其容易被“幸存者样本”美化。缓解：使用全历史可交易股票池，按当期可交易性动态筛选。citeturn15search19turn21search1  
- **停牌/缺失值与指标断裂**：停牌会破坏周序列等间隔假设，TimeSeriesSplit要求样本等间隔更可比；你需要先补齐周索引并用“是否停牌/是否成交”做辅助特征或过滤。citeturn1search3turn21search0  
- **过拟合与数据挖掘偏误**：技术指标空间巨大，“选到一个看似很强的组合”很容易。建议：  
  - 用Reality Check/SPA控制多重尝试；  
  - 用DSR/PBO评估“回测胜者诅咒”。citeturn15search19turn18search1turn16search0turn16search1  
- **宏观与行业轮动导致的非平稳**：周频策略更暴露于“状态切换”（政策/流动性/行业轮动），建议做：分市场状态/分行业/分波动区间的分层回测，并用滚动训练更新SVM。citeturn17search2turn18search3  

## 附录：示例代码与图表示例

以下代码以“周五收盘形成特征、预测下周方向、周一开盘交易”为例，展示数据处理、SVM训练、时间序列CV与回测框架骨架。预处理与模型通过Pipeline组合，遵循“先切分、后拟合预处理”的防泄漏原则。citeturn21search1turn1search3turn22search3  

### 周线聚合与标签构造（pandas示例）
```python
import pandas as pd
import numpy as np

def to_weekly_ohlcv(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    df_daily index: DatetimeIndex
    columns: open, high, low, close, volume
    """
    # 确保按时间排序
    df_daily = df_daily.sort_index()

    # 周线聚合：以周五/最后交易日为周末
    w = pd.DataFrame({
        "open":  df_daily["open"].resample("W-FRI").first(),
        "high":  df_daily["high"].resample("W-FRI").max(),
        "low":   df_daily["low"].resample("W-FRI").min(),
        "close": df_daily["close"].resample("W-FRI").last(),
        "volume":df_daily["volume"].resample("W-FRI").sum(),
    })

    # 去除全缺失周（例如全周停牌）
    w = w.dropna(subset=["close"])
    return w

def make_weekly_label(w: pd.DataFrame, horizon_weeks: int = 1) -> pd.Series:
    """
    方向标签：预测未来horizon_weeks周收益是否为正
    """
    fwd_ret = w["close"].shift(-horizon_weeks) / w["close"] - 1.0
    y = (fwd_ret > 0).astype(int)
    return y
```

### 技术指标计算（示意：用“公式接口/平台函数/自编函数”皆可）
说明：你可以直接用数据平台（同花顺/米筐）已有指标，也可按其公开公式自算。下例只示范几个核心指标的“特征形态”，并鼓励把Level转为Ratio/Delta/Z-score。citeturn20view0turn12view0  

```python
def add_features(w: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=w.index)

    # 基础收益与区间
    X["ret_1w"] = w["close"].pct_change(1)
    X["range_hl"] = (w["high"] - w["low"]) / w["close"]

    # 例：MA比值（需要你实现ma函数或调用平台/库）
    X["close_ma_12w"] = w["close"] / w["close"].rolling(12).mean() - 1

    # 例：波动率(收益标准差)
    X["ret_std_26w"] = X["ret_1w"].rolling(26).std()

    # 例：成交量log与变化
    X["log_vol"] = np.log1p(w["volume"])
    X["dlog_vol_1w"] = X["log_vol"].diff(1)

    # 对“累计型”指标（如OBV/PVT）建议只放变化率/差分
    # X["obv"] = ...
    # X["dobv_1w"] = X["obv"].diff(1)

    # 训练时再统一做缺失值处理与缩放
    return X
```

### SVM训练 + 时间序列CV + 调参（scikit-learn）
```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

def train_svm_time_series_cv(X, y):
    # 时间序列切分：严格时间顺序
    tscv = TimeSeriesSplit(n_splits=5)

    pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", gamma="scale", class_weight="balanced"))
    ])

    param_grid = {
        "clf__C": [0.1, 1, 10, 100],
        "clf__gamma": [1e-3, 1e-2, 1e-1, "scale"],
    }

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=tscv,
        scoring="roc_auc",   # 或 balanced_accuracy / f1
        n_jobs=-1,
        refit=True
    )
    gs.fit(X, y)
    return gs
```

### 周频回测框架骨架（含成本与成交约束接口）
成本项建议参数化（印花税/经手费/过户费/佣金来自公开口径；滑点与涨跌停成交失败需要你自己建模）。citeturn2view0turn10search4turn9search10turn9search4turn6view1turn6view2  

```python
def backtest_weekly(w: pd.DataFrame, prob_up: pd.Series,
                    buy_threshold=0.55, sell_threshold=0.45,
                    commission=0.0003,      # 万3示例：可调
                    stamp_duty=0.0005,       # 卖出单边：政策口径常用0.05%模拟
                    transfer_fee=0.00001,    # 0.01‰=0.00001
                    handling_fee=0.0000341,  # 0.00341%=0.0000341
                    slippage=0.0002):        # 自设：例如2bps
    """
    简化：只做多/空仓（或全仓/空仓），每周一开盘换仓。
    实盘中需加入：涨跌停无法成交、停牌、最小交易单位、资金占用等。
    """
    idx = w.index
    position = 0  # 1=持仓, 0=空仓
    equity = 1.0
    equity_curve = []

    for t in range(len(idx)-1):
        dt = idx[t]
        next_dt = idx[t+1]

        # 生成交易信号（用dt周收盘信息预测下一周）
        p = prob_up.loc[dt]
        target_pos = position
        if p >= buy_threshold:
            target_pos = 1
        elif p <= sell_threshold:
            target_pos = 0

        # 执行换仓：假设在next_dt开盘成交（需要用下周开盘价；此处用close近似示意）
        if target_pos != position:
            # 买入成本
            if target_pos == 1:
                cost = commission + transfer_fee + handling_fee + slippage
            # 卖出成本（含印花税）
            else:
                cost = commission + transfer_fee + handling_fee + stamp_duty + slippage
            equity *= (1 - cost)
            position = target_pos

        # 持有一周收益（示意用close-to-close周收益）
        r = w["close"].iloc[t+1] / w["close"].iloc[t] - 1.0
        equity *= (1 + position * r)
        equity_curve.append((next_dt, equity))

    return pd.Series({d:v for d,v in equity_curve}).sort_index()
```

### 绩效曲线示意图（matplotlib示例）
```python
import matplotlib.pyplot as plt

def plot_equity(equity_curve: pd.Series, title="Weekly Strategy Equity Curve"):
    plt.figure()
    plt.plot(equity_curve.index, equity_curve.values)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.show()
```

可立即执行的实验步骤（每条≤30字）  
1) 拉取日线并聚合周OHLCV  
2) 计算10组指标并做TS-CV调参  
3) 加入涨跌停与成本后回测对比