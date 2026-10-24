# Обзор исследований для двух заданий Т-Банка по рекомендательным системам

**Дата проверки: 24 октября 2026 года.** Использованы первоисточники: статьи, авторские репозитории, документация и официальные страницы публикаций. Эксперименты и код из репозиториев не запускались; файлы на диске не изменялись.

В обзор включены рецензируемые публикации и отдельно обозначенные препринты. Для нескольких найденных работ полный текст оказался недоступен — их результаты помечены как **не проверенные**. Реестр Yambda охватывает все обнаруженные в ходе поиска экспериментальные работы, но гарантировать библиографическую исчерпывающую полноту открытой индексации нельзя.

Ваши экспериментальные результаты принимаются как данные постановки; независимо они здесь не воспроизводились.

## 1. Главные выводы

1. **Опубликованные результаты Yambda нельзя напрямую сопоставлять с NDCG@10 = 0,764 из задания 1.** Оригинальный бенчмарк оценивает извлечение релевантных объектов из каталога, а вы ранжируете уже залогированные алгоритмические показы внутри пользователь-дня. Отличаются кандидаты, единица усреднения и смысл отрицательных примеров. [Ploshkin et al., §4](https://arxiv.org/html/2505.22238v2#S4)

2. **Для результата «трансформер отдельно слабее бустинга, как признак полезен» есть очень близкий индустриальный прецедент — ARGUS.** В сравнении с производственным GBDT авторы получают отрицательное относительное изменение pair accuracy для самостоятельного трансформера и положительное — при добавлении его score в бустинг. Это сильнее подходит для обсуждения вашего Perseus, чем сравнения архитектур на академическом next-item бенчмарке. [ARGUS, табл. 3–5](https://arxiv.org/html/2507.15994)

3. **Сильное правило повторов в музыке согласуется с литературой.** На другом датасете Яндекс Музыки Personalized Most Popular превосходит несколько последовательных трансформеров; добавление персональной частотности существенно улучшает их результаты. Это прямое обоснование вашего канала повторов, но не основание переносить опубликованные числа на Yambda. [Abbattista et al., §3–4, табл. 2](https://arxiv.org/html/2409.04329)

4. **Автоматический поиск агрегатов имеет устоявшиеся предшественники.** DFS/Featuretools ближе всего к вашей грамматике событий; OpenFE — к оценке добавочной полезности кандидатов; CAAFE — к агенту, который предлагает признаки и получает экспериментальную обратную связь. Содержательная часть вашей работы — временная корректность, пространство событийных агрегатов и проверенный прирост над сильным бейзлайном. [DFS](https://groups.csail.mit.edu/EVO-DesignOpt/groupWebSite/uploads/Site/DSAA_DSM_2015.pdf), [OpenFE](https://proceedings.mlr.press/v202/zhang23ay.html), [CAAFE](https://arxiv.org/html/2305.03403)

5. **Нашлась особенно близкая работа 2026 года на Yambda-5B: сессионная оптимизация REINFORCE с многошаговой off-policy коррекцией.** При этом её обучение использует награды из лога; модель обратной связи применяется для OPE и планирования при выдаче. Это существенное отличие от вашего обучения непосредственно против reward model. [Matveev et al., §3–4](https://arxiv.org/html/2607.02818)

6. **OneRec необходимо цитировать с указанием версии.** Ранняя статья использует итеративный DPO; Technical Report — ECPO, вариант GRPO, без KL-loss и с совместным SFT; OneRec-V2 переходит к GBPO и реальной пользовательской обратной связи. Нельзя описывать всё семейство одной формулой «DPO/GRPO с KL». [OneRec](https://arxiv.org/html/2502.18965), [Technical Report, §2.4](https://arxiv.org/html/2506.13695v4), [V2, §3](https://arxiv.org/html/2508.20900)

7. **Рост предсказанной награды недостаточен для вывода об улучшении рекомендаций или удержания.** Помимо общего результата Gao et al., есть прямые рекомендательные примеры: reward-only GRPO повышает оценки RM одновременно с ухудшением retrieval-метрик. Нужны отдельные кривые reward, KL, качества по отложенным действиям и концентрации выдачи. [Gao et al.](https://proceedings.mlr.press/v202/gao23h.html), [AdaGRPO, §6.2, табл. 4](https://arxiv.org/html/2606.08480)

8. **KION подходит для проверки переноса подхода, но плохо подходит для переноса именно механизма повторного потребления.** Публичный датасет оставляет последнюю запись пары пользователь–объект, а длительность и процент просмотра агрегирует по всем просмотрам. Исходную последовательность повторов и её исторические снимки восстановить нельзя. [KION, §2.2](https://arxiv.org/html/2209.00325)

## 2. Опубликованные числа на Yambda

### 2.1. Оригинальная статья Yambda: все запрошенные результаты 50M

**Источник:** Ploshkin et al., *Yambda-5B — A Large-Scale Multi-modal Dataset for Ranking And Retrieval*, 2025, [arXiv:2505.22238v2](https://arxiv.org/html/2505.22238v2). Данные и реализации: [Hugging Face](https://huggingface.co/datasets/yandex/yambda), [benchmarks](https://huggingface.co/datasets/yandex/yambda/tree/main/benchmarks).

**Протокол:** GTS, 300 дней обучения, разрыв 30 минут, один тестовый день; параметры и пользовательские состояния заморожены на начало теста. Listen+ определяется порогом 50% длительности. Для подбора гиперпараметров выделен день в обучении; SASRec не проходил такой подбор из-за вычислительных ограничений. Ограничение оценки только `is_organic=0` в описании этого бенчмарка не установлено; переносить на него вашу фильтрацию нельзя. [§4.1–4.3](https://arxiv.org/html/2505.22238v2#S4)

**Listen+, Yambda-50M — таблица 6:**

| Модель | NDCG@10 | NDCG@100 | Recall@10 | Recall@100 |
|---|---:|---:|---:|---:|
| Random | 0,0000 | 0,0000 | 0,0000 | 0,0002 |
| MostPop | 0,0186 | 0,0249 | 0,0064 | 0,0321 |
| DecayPop | 0,0260 | 0,0323 | 0,0122 | 0,0479 |
| ItemKNN | 0,0781 | 0,0934 | 0,0373 | 0,1297 |
| iALS | 0,0407 | 0,0642 | 0,0128 | 0,0808 |
| BPR | 0,0389 | 0,0641 | 0,0139 | 0,0836 |
| SANSA | 0,0069 | 0,0095 | 0,0031 | 0,0137 |
| SASRec | 0,0748 | 0,0764 | 0,0325 | 0,1026 |

**Like, Yambda-50M — таблица 7:**

| Модель | NDCG@10 | NDCG@100 | Recall@10 | Recall@100 |
|---|---:|---:|---:|---:|
| Random | 0,0000 | 0,0000 | 0,0000 | 0,0003 |
| MostPop | 0,0046 | 0,0097 | 0,0083 | 0,0222 |
| DecayPop | 0,0180 | 0,0269 | 0,0333 | 0,0651 |
| ItemKNN | 0,0125 | 0,0251 | 0,0199 | 0,0648 |
| iALS | 0,0078 | 0,0224 | 0,0103 | 0,0568 |
| BPR | 0,0056 | 0,0187 | 0,0073 | 0,0526 |
| SANSA | 0,0068 | 0,0203 | 0,0105 | 0,0616 |
| SASRec | 0,0100 | 0,0208 | 0,0175 | 0,0603 |

Обе таблицы переписаны из [таблиц 6–7 первоисточника](https://arxiv.org/html/2505.22238v2#S4).

Для остальных масштабов ключевые строки SASRec:

| Данные | Цель | NDCG@10 | NDCG@100 | Recall@10 | Recall@100 | Где |
|---|---|---:|---:|---:|---:|---|
| 500M | Listen+ | 0,0754 | 0,0884 | 0,0336 | 0,1240 | Табл. 6 |
| 5B | Listen+ | 0,0647 | 0,0847 | 0,0289 | 0,1214 | Табл. 6 |
| 500M | Like | 0,0125 | 0,0237 | 0,0200 | 0,0890 | Табл. 7 |
| 5B | Like | 0,0136 | 0,0348 | 0,0217 | 0,1026 | Табл. 7 |

### 2.2. Другие работы: проверенные результаты и границы протокола

Здесь приведены **ключевые результаты**, а не все строки всех статей. Значения в процентах явно обозначены. Неуточнённый авторами элемент протокола не восстанавливался предположением.

| Работа / данные | Протокол | Модель и опубликованные числа | Где |
|---|---|---|---|
| **SEATER replication**, 50M | Только органика; прослушивания не короче 50%; 299 дней train, 30 минут gap, день validation, день test | `NDCG@20 / NDCG@50 / Recall@20 / Recall@50`: GRU4Rec **0,0680 / 0,0820 / 0,0701 / 0,1203**; BERT4Rec **0,0412 / 0,0503 / 0,0448 / 0,0776**; SASRec **0,0960 / 0,1138 / 0,1006 / 0,1636**; SEATER **0,1245 / 0,1370 / 0,1175 / 0,1705** | [§4.1, табл. 4, с. 10](https://arxiv.org/pdf/2512.18434) |
| **SEATER: варианты построения дерева**, тот же протокол | Среднее по трём seeds | NDCG@20: Original **0,1245 ± 0,0073**, Hybrid **0,1495 ± 0,0210**, Greedy **0,1399 ± 0,0179**. Отдельная абляция порога 2000: **0,1616 ± 0,0009** — её нельзя подменять основным результатом | [Табл. 5–6, с. 11–12](https://arxiv.org/pdf/2512.18434) |
| **CREATE**, 50M, likes | Global temporal split; подбор на validation, затем переобучение train+validation. Конкретные границы основного split в §4.2 не заданы | SASRec → CREATE: NDCG@10 **1,11% → 1,54%**; Recall@10 **1,35% → 1,86%**; NDCG@100 **1,46% → 1,84%**; Recall@100 **3,15% → 3,40%** | [§4.1–4.2, табл. 1–2](https://arxiv.org/html/2602.23471) |
| **Position-Aware Sequential Attention**, Y-likes / Y-listens | 5-core; временные доли 95% / 2% / 3%; successive evaluation. Масштаб исходного варианта и порог Listen+ для listens не установлены | NDCG@10, Classic / RoPE / предложенный kernel: likes **0,0145 / 0,0148 / 0,0151**; listens **0,0095 / 0,0114 / 0,0123** | [§5.1–5.2, табл. 2](https://arxiv.org/html/2602.21052) |
| **Variable-Length Semantic IDs**, likes | После фильтрации 82M событий; последняя неделя test; одна тестовая интеракция на пользователя; оценка на 5% пользователей; бюджет истории 512 токенов | Recall@100: R-KMeans **0,1637**; fixed dVAE **0,1645**; varlen λ=1/2/3 **0,1676 / 0,1794 / 0,1820**. Coverage@100 для R-KMeans / varlen λ=3: **18,2% / 25,8%** | [§4.1.2, табл. 5](https://arxiv.org/html/2602.16375) |
| **Semantic ID Staleness**, «500M-like subsample» | После preprocessing 631 227 событий; десять последовательных блоков; base на 1–8, адаптация на 9, test на 10; seen не удаляются; десять seeds | LETTER FT-old → Hungarian alignment: Recall@100 **0,0450 → 0,0519**, Recall@500 **0,1116 → 0,1174**, NDCG@10 **0,0112 → 0,0125** | [§4.1, табл. 1–2](https://arxiv.org/html/2604.13273) |
| **ACARec**, 50M, холодные треки | Уникальные user–track пары; последние четыре недели — cold evaluation; новые items делятся 30/70 на validation/test; исключены listens <20%; только известные train пользователи и артисты | DeepMusic+ArtistMean → ACARec, Overall: Recall@20 **0,1665 → 0,1840**, NDCG@20 **0,1581 → 0,1745**. Discovery: **0,0995 → 0,1258**, **0,0604 → 0,0814** | [§4.1, табл. 2](https://arxiv.org/html/2604.07090) |
| **Ducho**, подвыборка 50M | 20-core и дополнительная выборка; итог 4 153 пользователя, 6 976 items; split 80/20, validation — 10% train. Временной характер split не установлен | `Recall@20 / NDCG@20`, **в процентах**: VBPR **3,67 / 2,31**; LATTICE **5,91 / 3,69**; FREEDOM **6,14 / 3,84**; BM3 **4,73 / 2,93**; NGCF-M **5,71 / 3,41**; PGL **6,62 / 4,09** | [§4.1, §4.4, табл. 4 и 14](https://arxiv.org/html/2409.15857v2) |
| **Session-Level REINFORCE**, 5B | Последняя неделя test; история chunks по 512 событий; non-organic targets; каталог оценки — top-500 000 train-треков плюс target. Positive: like либо полное прослушивание | `Recall@100 all / positive`: NIP-CE **0,3153 / 0,3217**; Positive-CE **0,3029 / 0,3298**; REINFORCE **0,3086 / 0,3174**. DR-оценки возврата: **0,6046 / 0,6184 / 0,6306**, соответственно | [§4.1, табл. 1 и 3](https://arxiv.org/html/2607.02818) |
| **Planning over MF-MDPs**, вариант Yambda не уточнён | Положительное прослушивание ≥80%; каталог top-5000; общие ALS-факторы; LLN и GTS; три seeds | LLN, Static → Plan-K: Recall@10 **0,0138 → 0,0179**, NDCG@10 **0,0112 → 0,0148**. GTS: Recall@10 **0,0306 → 0,0256**, NDCG@10 **0,0623 → 0,0628** | [§4.1, табл. 1](https://arxiv.org/html/2607.02115) |
| **Retrievers Need Verifiers**, 50M / 500M / 5B | Заявлен протокол Yambda; 20% пользователей выделены для выбора fusion weight; reranking top-1000; повторы разрешены; один seed | SASRec → +Verifier, `Recall@10 / Recall@100`, **в процентах**: 50M **5,172 / 12,08 → 5,292 / 12,83**; 500M **4,163 / 12,03 → 4,883 / 14,53**; 5B **3,520 / 10,52 → 4,410 / 14,46** | [§4.3–4.5, табл. 4](https://arxiv.org/html/2609.12270) |
| **CutBCE**, 50M | Последние восемь событий пользователя — test; отдельные предыдущие блоки — dev/train; full-vocabulary ranking. Hit@8 означает хотя бы одно совпадение top-8 с восемью targets | При batch 64, sequence 256: SASRec-BCE **Hit@8 = 0,186**, CutBCE **0,183**; память **22,37 → 7,67 GiB**; скорость **6,14 → 20,01 steps/s** | [§III, §IV-B, табл. III](https://arxiv.org/html/2610.05559) |

**Что здесь особенно важно:** даже внутри опубликованных результатов SASRec встречаются разные цели, каталоги и обновление истории. Разница между строками не является эффектом одного архитектурного изменения.

### 2.3. Что сделано в этих работах и какой код доступен

| Работа, статус | Содержание и польза для наших заданий | Код |
|---|---|---|
| **Efficient Optimization of Hierarchical Identifiers for Generative Recommendation**, Valeau et al., препринт 2025, версия 2026 | Репликация SEATER и ускорение построения иерархических IDs. Полезна как воспроизводимый generative retrieval baseline; органический протокол отличается от нашего | [re-seater](https://github.com/joshrosie/re-seater) |
| **Cross-Representation Knowledge Transfer for Improved Sequential Recommendations**, Gimranov et al., 2026, препринт | CREATE согласует последовательное и графовое представления. Можно заимствовать комбинацию локальной последовательности с глобальным collaborative-сигналом | [Авторская ссылка на код](https://anonymous.4open.science/r/multirepr_recsys-761F) |
| **Position-Aware Sequential Attention for Accurate Next Item Recommendations**, Nabiev, Frolov, 2026, препринт | Обучаемый позиционный kernel внутри attention. Полезен как проверка recency-индуктивного смещения, а не как объяснение всех преимуществ агрегатов | [anonymcode/IJCAI2026](https://github.com/anonymcode/IJCAI2026) |
| **Variable-Length Semantic IDs for Recommender Systems**, Khrylchenko, 2026, препринт | Переменная длина item-ID позволяет вместить больше исторических событий в фиксированный токенный бюджет. Это альтернатива огромному словарю треков GPT-2 | [varlen_semantic_ids](https://github.com/KhrylchenkoKirill/varlen_semantic_ids) |
| **Mitigating Collaborative Semantic ID Staleness in Generative Retrieval**, Baikalov et al., 2026 | Согласование обновлённых IDs со старым checkpoint при временном дрейфе. Заимствовать rolling-origin оценку и раздельное измерение качества/стоимости обновления | [semantic-id-alignment](https://github.com/iskbaga/semantic-id-alignment) |
| **Leveraging Artist Catalogs for Cold-Start Music Recommendation**, 2026 | ACARec восстанавливает представление нового трека через аудио и каталог артиста. Полезна для отдельной оценки новых артистов и новых треков знакомых артистов | [ACARec](https://github.com/gmeehan96/ACARec) |
| **Large-scale Benchmarks for Multimodal Recommendation with Ducho**, Attimonelli et al., версия 2026 | Сравнение мультимодальных моделей на общей инфраструктуре. Yambda-эксперимент относится к обновлённой версии; ранний arXiv-ID 2024 года не означает использование Yambda до её выпуска | [multimod-recs-bench-ducho](https://github.com/sisinflab/multimod-recs-bench-ducho) |
| **Session-Level Optimization for Large-Scale Retrieval using REINFORCE with Multi-Step Off-Policy Correction**, Matveev et al., 2026 | Сессионное off-policy обучение и планирование через feedback model. Ближайший найденный Yambda-аналог задания 2; публичная реализация в открытом тексте не найдена | [Статья](https://arxiv.org/abs/2607.02818) |
| **Planning over Matrix-Factorization MDPs for Candidate Generation**, Trapeznikov, Utushkin, 2026 | Планирование поверх фиксированных MF-представлений. Особенно полезен отрицательный результат на GTS: улучшение LLN не гарантирует переноса в будущее | [Статья](https://arxiv.org/abs/2607.02115); код не найден |
| **Recommendation Retrievers Need Verifiers: Universal Generative Reranking for Sequential Recommendations**, Zhang et al., 2026, препринт | Генеративный verifier переранжирует кандидатов retrieval-модели. Заимствовать раздельные repeat/new срезы и контролируемую комбинацию scores | [Статья](https://arxiv.org/abs/2609.12270); код не найден |
| **Cut Binary Cross Entropy: Efficient Large-Vocabulary Loss and Gradient Kernels for Sequential Recommendation**, Li et al., 2026, препринт | Вычислительно эффективный full-vocabulary BCE. Полезен при ограничении памяти; его Hit@8 не является вашим NDCG@10 | [Оператор в RecML](https://github.com/AI-Hypercomputer/RecML/blob/main/recml/core/ops/binary_cross_entropy_ops.py) |

Дополнительно обнаружены:

- **Jiang, Yang, *Efficient Swing Computation for Retrieval in Large-Scale Recommender Systems*, 2026, [arXiv:2609.16850](https://arxiv.org/html/2609.16850).** Yambda используется как большой двудольный граф для проверки вычисления Swing. В §6.3 сообщается точность восстановления top-K точного алгоритма **выше 99,97%**; это **не precision рекомендаций по пользовательской релевантности**. Полезно для быстрого item-to-item канала; [код K-ASC](https://github.com/HKBU-LAGAS/K-ASC).

- **MLCommons, *DLRMv4: The Next-Generation Recommendation MLPerf Training Benchmark*, 2026.** Открытый инженерный бенчмарк HSTU на Yambda-5B: Listen+ против skip, история перед каждым событием, временной holdout. **AUC ≥0,75 — целевой порог сходимости**, не GAUC пользователь-дня и не сравнительный SOTA-результат. См. [описание](https://mlcommons.org/2026/10/dlrmv4-training/) и [README, §3 и §5](https://github.com/mlcommons/training/tree/master/recommendation).

- **De Toni et al., *Towards Mitigating Unwanted Recommendations with Risk-Controlling Recommender Systems*, 2026, [DOI:10.1145/3816736](https://doi.org/10.1145/3816736).** Публикация подтверждена [университетской карточкой](https://iris.unitn.it/handle/11572/494612); [авторский репозиторий](https://github.com/geektoni/risk-controlling-recommender-systems) содержит Yambda-конфигурацию. Полный текст через ACM недоступен: **числа и протокол статьи не проверены**. Потенциально релевантна контролю нежелательных рекомендаций.

- **Klenitskiy et al., *An Analysis of Sequential Patterns in Datasets for Evaluation of Sequential Recommendations*, 2026, [DOI:10.1145/3787969](https://doi.org/10.1145/3787969).** Подтверждены [публикация](https://publications.hse.ru/en/articles/1122974633) и [код](https://github.com/Antondfger/Does-It-Look-Sequential). Полный текст расширенной версии не открылся; **экспериментальные числа на Yambda не проверены**. Не следует приписывать их доступной [статье 2024 года](https://arxiv.org/abs/2408.12008), где Yambda ещё нет.

- **Perseus:** открытые эксперименты включают Yambda-50M/500M; подробности — в следующем разделе. **Числа из самой demo-статьи не проверены.**

**Не включены в Yambda-бенчмарки:** ARGUS и *Correcting the LogQ Correction*. У ARGUS явно описан другой, закрытый музыкальный датасет; найденный полный текст второй работы также не содержит Yambda-экспериментов. [ARGUS, §4.1.1](https://arxiv.org/html/2507.15994), [LogQ, §4](https://arxiv.org/html/2507.09331)

## 3. Работы для задания 1

### 3.1. Автоматическое построение признаков

| Работа | Что сделано | Чем релевантно и что заимствовать |
|---|---|---|
| **Kanter, Veeramachaneni. Deep Feature Synthesis: Towards Automating Data Science Endeavors**, 2015. [Первичный PDF](https://groups.csail.mit.edu/EVO-DesignOpt/groupWebSite/uploads/Site/DSAA_DSM_2015.pdf) | Композиция преобразований и агрегатов по связанным таблицам; демонстрации на поведенческих данных | Ближайший концептуальный предшественник вашей грамматики. Описывать пространство операторов, глубину композиции и ограничения вычисления |
| **Featuretools**. [Код](https://github.com/alteryx/featuretools), [документация времени](https://featuretools.alteryx.com/en/stable/getting_started/handling_time.html) | Практическая реализация DFS с time index, cutoff time и training window | Заимствовать явную модель «когда значение стало известно». Для снимка начала дня отдельно фиксировать включение/исключение самой границы |
| **Zhang et al. OpenFE: Automated Feature Generation with Expert-level Performance**, ICML 2023. [Статья](https://proceedings.mlr.press/v202/zhang23ay.html), [код](https://github.com/IIIS-Li-Group/OpenFE) | Feature boosting оценивает добавочную полезность кандидатов; двухэтапное прореживание уменьшает поиск | Подходит для отбора поверх уже сильных ручных признаков. Сам по себе не решает корректное построение исторических окон из сырых логов |
| **Horn, Pack, Rieger. The autofeat Python Library for Automated Feature Engineering and Selection**, 2019/2020. [arXiv:1901.07329](https://arxiv.org/abs/1901.07329), [код](https://github.com/cod3licious/autofeat) | Генерация нелинейных комбинаций и устойчивый отбор для интерпретируемых линейных моделей | Полезен как классический AutoFE baseline. Поиск алгебраических преобразований отличается от поиска ключей, фильтров и временных агрегатов |
| **Hollmann, Müller, Hutter. Large Language Models for Automated Data Science: Introducing CAAFE for Context-Aware Automated Feature Engineering**, 2023. [arXiv:2305.03403](https://arxiv.org/html/2305.03403), [код](https://github.com/noahho/CAAFE) | LLM итеративно генерирует Python-признаки и объяснения, получает оценку downstream-модели | Прямой предшественник агента-гипотезогенератора. Сравнивать с грамматикой при одинаковом бюджете и закрытом test; опубликованные эксперименты — табличные, не Yambda |
| **Thakur et al. Towards Reliable ML Feature Engineering via Planning in Constrained-Topology of LLM Agents**, 2026, препринт. [arXiv:2601.10820](https://arxiv.org/html/2601.10820) | Агентный workflow создаёт взаимозависимые конфигурации и PySpark-код для production feature engineering | Релевантно надёжности выполнения гипотез. Вход уже содержит спецификацию признаков; качество workflow нельзя выдавать за доказательство лучшего поиска предикторов |

У CAAFE средний ROC AUC повышается **с 0,798 до 0,822** на рассмотренных авторами табличных задачах — **abstract и §5**. Это иллюстрация работоспособности подхода, не ориентир ожидаемого прироста GAUC на рекомендациях. [Первоисточник](https://arxiv.org/html/2305.03403)

**Наиболее точное тематическое совпадение, которое не удалось прочитать полностью:** Ye et al., *SAFE: LLM Agent for Sequential Feature Engineering in Industrial Recommendation*, 2026, [DOI:10.1007/s11704-026-60956-5](https://journal.hep.com.cn/fcs/EN/10.1007/s11704-026-60956-5). На странице издателя подтверждены название, авторы и принятие; полный текст закрыт. **Метод, результаты и код не проверены.** Его нельзя содержательно пересказывать по одному названию.

**Моя рекомендация для позиционирования:** разделить два исследовательских вопроса:

- Может ли ограниченный поиск операторов находить полезные агрегаты сверх экспертного набора?
- Даёт ли семантическое предложение гипотез LLM преимущество над таким поиском при равных затратах?

Победа одного способа отбора после значительно большего числа попыток ещё не показывает преимущество генератора гипотез.

### 3.2. Бустинг, последовательные модели и гибриды

**Khrylchenko et al., ARGUS: *Scaling Recommender Transformers to One Billion Parameters*, 2025/2026, [arXiv:2507.15994](https://arxiv.org/html/2507.15994).**

Обучение объединяет next-item и feedback prediction, затем следует impression-level ranking fine-tuning. База сравнения — производственный GBDT с ручными признаками и выходами прежних нейромоделей; данные закрытые.

| Эксперимент ARGUS | Самостоятельный score | Score как признак GBDT | Где |
|---|---:|---:|---|
| Medium, контекст 2048 | −5,33% | +2,32% | Табл. 3 |
| Large, около миллиарда параметров | −4,78% | +2,66% | Табл. 3 |
| Medium, контекст 8192 | −4,73% | +2,77% | Табл. 5 |

Это **относительные изменения pair accuracy**, не абсолютные пункты GAUC. Таблица 6 отдельно сообщает онлайн-эффекты последовательных внедрений; смешивать их с офлайн-абляциями нельзя. [§4.1–4.6](https://arxiv.org/html/2507.15994)

**Что позаимствовать:** сравнение трёх вариантов — ручные признаки, нейросигнал отдельно, гибрид; анализ длины истории; единый момент обновления пользовательского состояния.

**KION Challenge** даёт второй индустриальный пример: все лучшие решения используют финальный GBDT, а нейронные рекомендатели поставляют часть кандидатных scores. Это описание сильных систем, а не чистая контролируемая абляция «нейросеть против бустинга». Числа — ниже. [KION, §3.3](https://arxiv.org/html/2209.00325)

### 3.3. Perseus Т-Банка

**Babkin et al., *Perseus: A Demo of Modular Personalization over Heterogeneous Event Sequences*, RecSys 2026, [DOI:10.1145/3773078.3841295](https://doi.org/10.1145/3773078.3841295).**

Подтверждены официальная [demo-публикация](https://recsys.acm.org/recsys26/demo-presentations/), [основная библиотека](https://github.com/perseus2026/perseus) и [perseus-exps](https://github.com/perseus2026/perseus-exps). Perseus — модульный фреймворк работы с разнородными событиями и задачами; конкретный результат определяется выбранной конфигурацией.

В открытом экспериментальном коде проверено:

- Подготовка Yambda-50M строит основную задачу на **likes**; listens фильтруются условием **`played_ratio_pct > 50`**.
- В `prepare_basis` по умолчанию используется **80/20 global timestamp split**.
- Реализована оценка Recall, NDCG и coverage; README предусматривает варианты с `mask_seen`.
- Поэтому опубликованный экспериментальный пакет **не совпадает** с вашим ранжированием `is_organic=0` и Listen+ внутри дня. [Подготовка Yambda](https://github.com/perseus2026/perseus-exps/blob/main/data_preprocessing/yambda-50m.py), [split](https://github.com/perseus2026/perseus-exps/blob/main/data_preprocessing/utils.py), [оценщик](https://github.com/perseus2026/perseus-exps/blob/main/eval_utils.py)

Полный текст demo-статьи через ACM не открылся: **её сравнительные числа не проверены**. Нельзя приписывать статье ваши −0,035/+0,004 GAUC или числа из вторичных пересказов.

Для отчёта полезно указать backbone, loss, длину и типы истории, candidate-aware или dot-product scoring, частоту обновления состояния. Формулировка «Perseus хуже бустинга» без этих условий слишком широка.

### 3.4. KION: проверенные протоколы и числа

**Petrov et al., *MTS Kion Implicit Contextualised Sequential Dataset for Movie Recommendation*, [arXiv:2209.00325](https://arxiv.org/html/2209.00325).** Данные: [KION_DATASET](https://github.com/irsafilo/KION_DATASET), [RecTools](https://github.com/MobileTeleSystems/RecTools).

#### Challenge

Предсказание просмотров **следующей закрытой недели**, включая холодных пользователей. Основная метрика — MAP@10.

| Решение | MAP@10 | Где |
|---|---:|---|
| Lashinin: neural + non-neural ensemble, LightGBM | 0,1221 | Табл. 3 |
| Petrov: neural + non-neural ensemble, LightGBM | 0,1214 | Табл. 3 |
| Zimin: non-neural ensemble, CatBoost | 0,1213 | Табл. 3 |
| Tikhonovich: ItemKNN + CatBoost | 0,1148 | Табл. 3 |
| Olga: popularity + CatBoost | 0,1135 | Табл. 3 |
| Popularity baseline | 0,0910 | Табл. 3 |

**Польза:** обоснование сильного гибрида из collaborative/sequence scores, агрегатов и GBDT. Эти результаты не сравниваются напрямую с локальным holdout публичного KION. [§3.1–3.3](https://arxiv.org/html/2209.00325)

#### eSASRec / RecTools

**Tikhonovich et al., *eSASRec: Enhancing Transformer-based Recommendations in a Modular Fashion*, 2025, [arXiv:2508.06450](https://arxiv.org/html/2508.06450), [код](https://github.com/blondered/transformer_benchmark).**

Протокол KION: 2-core пользователей, 5-core items; последние **14 дней** — test, холодные user/item из test удаляются; validation создаётся LOO из train. Цель сравнения — качество и охват каталога.

| Модель | NDCG@10 | Coverage@10 | Где |
|---|---:|---:|---|
| Popular-7-days | 0,1420 | 0,0035 | Табл. 2 |
| SASRec Vanilla, BCE, один negative | 0,1389 | 0,1130 | Табл. 2 |
| SASRec + Sampled Softmax | 0,1497 | 0,0785 | Табл. 2 |
| HSTU | 0,1694 | 0,2110 | Табл. 2 |
| eSASRec: LiGR + Sampled Softmax | 0,1657 | 0,3003 | Табл. 2 |
| SASRec + LiGR + gBCE-0.75 | 0,1685 | 0,2381 | Табл. 2 |

Заимствовать **сильный настроенный последовательный baseline** и accuracy–coverage анализ: проигрыш vanilla SASRec не доказывает проигрыш всего класса последовательных моделей. [§4.1–4.2](https://arxiv.org/html/2508.06450)

#### Положительная и отрицательная обратная связь

**Ivanova et al., *Benefiting from Negative yet Informative Feedback by Contrasting Opposing Sequential Patterns*, RecSys 2025, [arXiv:2508.14786](https://arxiv.org/html/2508.14786), [PNFRec](https://github.com/Veronika-Ivanova/pnfrec).**

KION: 5-core; порог отрицательного просмотра — **15%**; train до глобальной границы 90% событий; последующие пользователи разделены между validation/test, последняя интеракция — target.

Таблица 2, среднее пяти запусков:

| Модель | NDCG положительных targets @10 | ΔNDCG@10: положительные минус отрицательные |
|---|---:|---:|
| SASRec | 0,1316 | 0,0521 |
| PNFRec | 0,1310 | 0,0700 |

Здесь улучшение связано прежде всего с подавлением отрицательных объектов; положительная NDCG сама по себе не растёт. Полезно для вашей многоголовой reward model и оценки skip/dislike отдельно от Listen+. [§4–5](https://arxiv.org/html/2508.14786)

#### Документационный бенчмарк iALS

[RecTools v0.4.1: iALS with features](https://rectools.readthedocs.io/en/v0.4.1/examples/5_benchmark_iALS_with_features.html) — **пример документации, не отдельная исследовательская статья**. Использованы недельный holdout, фильтрация train по длительности просмотра и `filter_viewed=True`. Лучший показанный вариант с признаками имеет **MAP@10 = 0,076887**, таблица вывода ячейки `[244]`.

Это пригодный воспроизводимый технический ориентир, но его нельзя смешивать с private challenge MAP или eSASRec NDCG.

## 4. Работы для задания 2

### 4.1. Next-item модель, reward model и preference/RL alignment

**OneRec, Deng et al., 2025 — [arXiv:2502.18965](https://arxiv.org/html/2502.18965).**  
Генеративный encoder–decoder рекомендует последовательности item IDs; RM оценивает сгенерированные варианты, из которых строятся предпочтительные/непредпочтительные пары для итеративного DPO — §3.3. Заимствовать сравнение RL с preference optimization на тех же RM и кандидатах. Полная производственная реализация в открытой статье не найдена.

**OneRec Technical Report, 2025 — [arXiv:2506.13695v4](https://arxiv.org/html/2506.13695v4).**  
ECPO использует групповые нормированные advantages и P-Score reward model. **В §2.4.1 авторы явно удаляют KL-loss**, поскольку RL обучается совместно с SFT. Для нас полезны абляции группового размера, способа генерации и регуляризации; это не эквивалент вашего RLOO с точным KL.

**OneRec-V2 Technical Report, 2025 — [arXiv:2508.20900v4](https://arxiv.org/html/2508.20900).**  
Вводятся GBPO, duration-aware reward и обучение на реальном feedback; §3.2 обсуждает ограничения RM. Полезно как аргумент в пользу проверки реальными исходами. Не следует считать RM-score достаточной конечной метрикой.

**Chen et al., *On Softmax Direct Preference Optimization for Recommendation*, 2024 — [arXiv:2406.09215](https://arxiv.org/html/2406.09215), [S-DPO](https://github.com/chenyuxin1999/S-DPO).**  
S-DPO учитывает несколько negatives через модель частичных предпочтений и softmax-подобный objective. Это alignment LM-рекомендателя **без обязательной отдельной RM и без policy-gradient этапа**. Заимствовать multi-negative DPO baseline; учитывать, что непоказанный item не обязательно хуже положительного.

**Wang et al., *GFlowGR: Fine-tuning Generative Recommendation Frameworks with Generative Flow Networks*, препринт 2025 / SIGIR 2026 — [arXiv:2506.16114v3](https://arxiv.org/html/2506.16114), [код](https://github.com/Applied-Machine-Learning-Lab/SIGIR26_GFlowGR).**  
Генерация рассматривается как построение множества; behavior-aware RM задаёт utility, GFlowNet согласует с ней распределение генераций. Полезен как альтернативный способ получать несколько качественных и разнообразных кандидатов. Это другой objective, не RLOO.

**Matveev et al., 2026 — [Session-Level REINFORCE](https://arxiv.org/html/2607.02818).**  
Для вашей постановки важны feedback heads played ratio/like и явное моделирование продолжений. Награда — `like + 0,1 × ratio`; RL обучается по логам с оценённой behavior policy. Заимствовать сопоставление next-item, positive-only и session-return objectives. Авторы отдельно предупреждают: выбор rollout и оценка одной feedback model не дают независимого подтверждения качества — §4.3.

### 4.2. RLOO, GRPO и закрытая форма оптимума

**Ahmadian et al., *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs*, 2024 — [arXiv:2402.14740](https://arxiv.org/html/2402.14740).**  
§2.3 задаёт REINFORCE Leave-One-Out: для каждой независимо сэмплированной генерации baseline считается по остальным генерациям того же контекста. Это основной источник для вашего оптимизатора. Эксперименты относятся к LLM alignment; перенос эффективности на музыку нужно доказывать отдельно.

**Zhu et al., *Rank-GRPO: Training LLM-based Conversational Recommender Systems with Reinforcement Learning*, 2025/2026 — [arXiv:2510.20150](https://arxiv.org/html/2510.20150), [код](https://github.com/yaochenzhu/Rank-GRPO).**  
Единицей credit assignment становится позиция рекомендательного списка, а не отдельный текстовый токен или весь список. Полезно для многопозиционной выдачи; экспериментальная задача — conversational recommendation, не музыкальные track tokens.

**Xu et al., *Adaptive Loss Balancing for Noise-Robust GRPO in Generative Recommendation*, 2026, препринт — [arXiv:2606.08480](https://arxiv.org/html/2606.08480).**  
AdaGRPO оставляет SFT-обучение основой и допускает RL-обновление для примеров, где политика испытывает трудности, а RM различает релевантные и нерелевантные варианты. Заимствовать контроль доверия к RM и гибридную регуляризацию. Публичный код не найден; шаблонные conference-метаданные препринта не использую как подтверждение принятия.

**Точного опубликованного применения RLOO с вашим музыкальным протоколом на Yambda в найденных первоисточниках нет.** Найденный Yambda REINFORCE использует другую схему обучения.

**Rafailov et al., *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, NeurIPS 2023 — [arXiv:2305.18290](https://arxiv.org/html/2305.18290).**  
§4 и Appendix A.1 дают связь KL-регуляризованной оптимизации награды с распределением:

\[
\pi^*(a\mid x)
=
\frac{\pi_{\mathrm{ref}}(a\mid x)\exp(r(x,a)/\beta)}
{\sum_b\pi_{\mathrm{ref}}(b\mid x)\exp(r(x,b)/\beta)}.
\]

**Применение к вашему эксперименту — моя рекомендация:** при фиксированном контексте, фиксированной награде и одном действии это основной аналитический контроль. Можно непосредственно использовать logits  
`log π_ref + r / β`.

Для той же одношаговой задачи:

\[
J(\pi^*)-J(\pi)=
\beta\,D_{\mathrm{KL}}(\pi\|\pi^*).
\]

Следовательно, расстояние RLOO-политики до аналитического решения измеряет качество оптимизации заданной RM-цели. Оно **не измеряет истинность RM**. При изменяющихся состояниях и многошаговой награде подстановка только immediate reward уже не даёт общего оптимального решения сессионного MDP.

### 4.3. Переоптимизация reward model

**Gao, Schulman, Hilton, *Scaling Laws for Reward Model Overoptimization*, ICML 2023 — [первоисточник](https://proceedings.mlr.press/v202/gao23h.html), [arXiv:2210.10760](https://arxiv.org/abs/2210.10760).**  
В контролируемой постановке proxy RM оптимизируется через RL или best-of-N, а качество отслеживается отдельной gold RM. Усиление оптимизации proxy может ухудшать gold reward. Заимствовать независимого оценщика и анализ силы оптимизации; это не универсальная количественная закономерность для рекомендаций.

**Прямой рекомендательный пример — AdaGRPO.** На закрытых e-commerce логах, train за неделю и test за две последующие недели, reward-only GRPO повышает ClkRwd@10 **0,1981 → 0,2158**, но снижает HR@10 **11,01% → 10,39%**; hallucination rate растёт **0,08% → 0,89%**. Это **финальные checkpoints**, §6.2, таблица 4, а не лучшие промежуточные результаты. [Первоисточник](https://arxiv.org/html/2606.08480)

Для вашего отчёта особенно нужны графики по шагам обучения:

- training RM reward и независимые метрики по отложенному логу;
- точный KL, entropy, coverage и концентрация;
- награда и качество отдельно для repeat/new items;
- чувствительность к коэффициентам reward heads и β.

### 4.4. Повторное потребление

**Ren et al., *RepeatNet: A Repeat Aware Neural Recommendation Machine for Session-based Recommendation*, препринт 2018 / AAAI 2019 — [arXiv:1812.02646](https://arxiv.org/html/1812.02646), [код](https://github.com/PengjieRen/RepeatNet).**  
Обучаемый gate смешивает repeat decoder по истории и explore decoder. Это прямой архитектурный предшественник вашего канала повторов. Важно: повтор внутри текущей сессии и повтор за всю пользовательскую историю — разные определения.

**Abbattista et al., *Enhancing Sequential Music Recommendation with Personalized Popularity Awareness*, RecSys 2024 — [arXiv:2409.04329](https://arxiv.org/html/2409.04329), [код](https://github.com/sisinflab/personalized-popularity-awareness).**  
На **Yandex Music Event 2019, не Yambda**, после выборки каталога и GTS с test/validation долями по 0,1: NDCG@10 Personalized Most Popular **0,1826**, SASRec **0,1030**, SASRec+PPS **0,1647** — §3, таблица 2. Заимствовать частотный prior в score и обучение отклонения от него.

**Anderson et al., *The Dynamics of Repeat Consumption*, WWW 2014 — [DOI:10.1145/2566486.2568018](https://doi.org/10.1145/2566486.2568018), [авторский PDF](https://www.cs.utoronto.ca/~ashton/pubs/repeat-consumption-www2014.pdf).**  
Исследуются recency и привлекательность уже потреблённых объектов; вклады прошлых потреблений моделируются через затухание. Это обоснование для ваших частот, давности и decay-агрегатов. Работа условна на выборе повтора и не решает полностью выбор repeat против exploration.

**Li, Huang, de Rijke, *Repetition and Exploration in Offline Reinforcement Learning-based Recommendations*, 2023 — [авторский PDF](https://betsyhj.github.io/publication/drl4ir_cikm2023/drl4ir_cikm2023.pdf).**  
Разбирается поведение offline RL в отношении повторения и исследования новых объектов. Полезна для постановки отдельной оценки repeat/explore вместо одной общей accuracy. Публичную реализацию в проверенном источнике не установил.

**Lashinin et al., *SAFERec: Self-Attention and Frequency Enriched Model for Next Basket Recommendation*, 2024/2025 — [arXiv:2412.14302](https://arxiv.org/html/2412.14302), [код](https://github.com/anon-ecir-nbr/SAFERec).**  
Работа Т-Банка объединяет трансформер и явный frequency module для повторяющихся покупок. Это полезный авторский и методический ориентир для вашего repeat-канала, хотя домен — корзины и обучение supervised. **SAFERec не следует путать с SAFE об LLM feature engineering.**

Дополнительное подтверждение на Yambda: в *Retrievers Need Verifiers* доля повторных targets в варианте 500M равна **70,9%**, §4.5; это число относится к **их** протоколу и не подменяет вашу оценку доли повторов. [Первоисточник](https://arxiv.org/html/2609.12270)

### 4.5. Долгосрочные цели и удержание

**Cai et al., *Reinforcing User Retention in a Billion Scale Short Video Recommender System*, WWW 2023 — [arXiv:2302.01724](https://arxiv.org/html/2302.01724).**  
RLUR оптимизирует интервалы между сессиями; действие выбирает веса ансамблирования ранжирующих scores. Рассматриваются delayed reward, uncertainty и bias. Заимствовать определение наблюдаемого retention outcome и работу с задержкой; это другая action space, не непосредственная генерация треков. Публичный код не установлен.

**Zou et al., *Reinforcement Learning to Optimize Long-term User Engagement in Recommender Systems*, KDD 2019 — [arXiv:1902.05570](https://arxiv.org/html/1902.05570), [DOI:10.1145/3292500.3330668](https://doi.org/10.1145/3292500.3330668).**  
FeedRec использует Q-network и модель среды; reward включает глубину потребления и время возврата. Полезен как ранний пример явного различения immediate engagement и возвращаемости. Его симуляционную оценку нельзя считать эквивалентом реального A/B.

**Silva, Parra, Toro Icarte, *On the Unexpected Effectiveness of Reinforcement Learning for Sequential Recommendation*, ICML 2024 — [первоисточник](https://proceedings.mlr.press/v235/silva24b.html).**  
Авторы показывают, что часть улучшения next-item качества от вспомогательного RL-loss воспроизводится простым auxiliary objective, обучающим представление истории. Заимствовать контроль без RL: improved NDCG может объясняться представлением, а не долгосрочным планированием.

**Для нашей постановки:** прогноз «продолжит текущую сессию» и прогноз «вернётся завтра» — разные цели. «Освоение нового артиста» требует заранее заданных горизонта и критерия последующих повторных взаимодействий; одно первое прослушивание этого не устанавливает.

### 4.6. OPE без залогированных propensity

**Оценивать политику без известных propensity возможно при дополнительных предположениях. Получить безусловную причинную оценку из произвольного наблюдательного лога нельзя.**

- **Hao et al., *Bootstrapping Fitted Q-Evaluation for Off-Policy Inference*, ICML 2021 — [статья](https://proceedings.mlr.press/v139/hao21b.html), [PDF](https://proceedings.mlr.press/v139/hao21b/hao21b.pdf).** FQE оценивает value по переходам, не требуя известных вероятностей behavior policy. Заимствовать независимое обучение evaluator и bootstrap эпизодами; теоретические гарантии зависят от модельных предположений.

- **Uehara, Huang, Jiang, *Minimax Weight and Q-Function Learning for Off-Policy Evaluation*, ICML 2020 — [статья](https://proceedings.mlr.press/v119/uehara20a.html), [PDF](https://proceedings.mlr.press/v119/uehara20a/uehara20a.pdf).** MWL учит отношения state–action распределений без знания behavior policy, MQL — Q-функцию; они комбинируются в doubly robust схеме. Заимствовать альтернативу явному propensity model, сохраняя требования к данным и аппроксимации.

- **Chen et al., *Top-K Off-Policy Correction for a REINFORCE Recommender System*, WSDM 2019 — [arXiv:1812.02353](https://arxiv.org/html/1812.02353), [DOI](https://doi.org/10.1145/3289600.3290999).** Производственный YouTube REINFORCE с оценкой behavior policy и поправкой на выдачу нескольких items. Полезен как источник off-policy обучения; это не доказательство корректности propensity, восстановленных только из Yambda.

Для вашего лога:

- `is_organic` обозначает происхождение действия, **не вероятность показа**.
- Behavior cloning по доступной истории не обязательно восстанавливает реальную вероятность старой системы: могли отсутствовать candidate set, контекст и внутренние признаки.
- Оценка той же RM, против которой обучали политику, — проверка внутренней цели, а не независимая OPE.
- Если для сгенерированного трека нет достаточного покрытия в данных, его высокая предсказанная награда может быть экстраполяцией.

### 4.7. Несколько шагов вперёд и exposure bias

**Petrov, Macdonald, *Aligning GPTRec with Beyond-Accuracy Goals with Reinforcement Learning*, 2024 — [arXiv:2403.04875](https://arxiv.org/html/2403.04875).**  
GPTRec использует GPT-2-подобный decoder и PPO для авторегрессионной генерации списка; награды вычисляются по accuracy, diversity и popularity. **Отдельная обучаемая RM не обязательна:** value model предсказывает возврат, но это другая роль. Заимствовать list-level RL и сравнение с reranking; реализация опирается на [aprec/BERT4Rec_repro](https://github.com/asash/BERT4Rec_repro).

**Volodkevich et al., *Autoregressive Generation Strategies for Top-K Sequential Recommendations*, 2024/2025 — [arXiv:2409.17730](https://arxiv.org/html/2409.17730), [код](https://github.com/dalibra/autoregressive-generation-strategies-longterm-seqrec).**  
GPT-2 обучается на next item; несколько сгенерированных будущих последовательностей агрегируются через RRA/RA. Полезен как baseline многопроходной генерации **без RL**, особенно для множества дневных targets.

**Yang et al., *GenPlugin: A Plug-and-Play Framework for Long-Tail Generative Recommendation with Exposure Bias Mitigation*, 2025, препринт — [arXiv:2507.03568](https://arxiv.org/html/2507.03568).**  
Предлагается вероятностная замена истинных ID-токенов предсказанными при обучении и retrieval augmentation. Заимствовать диагностику накопления ошибок. Здесь exposure bias возникает **внутри генерации многотокенного item-ID**; у вас один трек — один токен, поэтому непосредственный аналог возникает при генерации нескольких треков подряд. Публичный код в тексте не найден.

Следует различать:

1. **Generation exposure bias:** обучение на истинных префиксах, inference на собственных генерациях.
2. **Logged exposure bias:** feedback имеется только для объектов, выбранных старой системой.

Устранение первого не устраняет второй. Кроме того, дописывание рекомендованного трека в историю как будто прослушанного — допущение о реакции пользователя, которое требует модели обратной связи.

## 5. Кандидаты на «две статьи» из условия

Это **предположения по тематическому совпадению**, не восстановленные исходные ссылки. Без даты составления задания достоверно выбрать пару нельзя.

| Кандидат | Почему вероятен | Что не совпадает |
|---|---|---|
| **OneRec: Unifying Retrieve and Rank with Generative Recommender and Preference Alignment**, [2502.18965](https://arxiv.org/abs/2502.18965), либо [Technical Report 2506.13695](https://arxiv.org/abs/2506.13695) | Очень близкая схема: предварительно обученный генеративный рекомендатель, RM, alignment по пользовательским предпочтениям | Semantic IDs, encoder–decoder/MoE и DPO/ECPO вместо GPT-2 track tokens и RLOO |
| **Aligning GPTRec with Beyond-Accuracy Goals with Reinforcement Learning**, [2403.04875](https://arxiv.org/abs/2403.04875) | Наиболее буквальное совпадение по GPT-подобному рекомендателю и RL-дообучению под несколько целей | Награда вычисляется по метрикам; critic не равен отдельной поведенческой RM |
| **GFlowGR**, [2506.16114](https://arxiv.org/abs/2506.16114) | Генеративная модель, behavior-aware RM и дообучение ради utility/разнообразного набора | GFlowNet objective вместо REINFORCE; конкретное содержание зависит от версии |
| **Session-Level Optimization… REINFORCE**, [2607.02818](https://arxiv.org/abs/2607.02818) | Совпадают музыка, Yambda, feedback model, REINFORCE и интерес к сессионной награде | Может быть исходной ссылкой только для задания, составленного после июля 2026; RM не является непосредственным источником train reward |

**Если задание было составлено до июля 2026 года**, наиболее правдоподобная пара из найденного — **OneRec + GPTRec**; GFlowGR остаётся сильным альтернативным кандидатом.

**Проверка Т-Банка и Яндекса:**

- У Lashinin/Ananyeva проверен **SAFERec**, но там supervised frequency-aware next-basket recommendation, без RM+RL. **Perseus** также не является прямым совпадением.
- Найден авторский инженерный материал Т-Банка [«RL RecSys в проде: хайп или игра вдолгую»](https://habr.com/en/companies/tbank/articles/882478/), обсуждающий SASRec+RL и использование RL-score в бустинге. Это полезный контекст, **не найденная исследовательская статья с описанной в задании схемой**.
- **ARGUS проверен:** next-item и feedback prediction присутствуют, но основной опубликованный pipeline — supervised pre-training и ranking fine-tuning. В проверенном тексте нет центрального этапа policy-gradient обучения против RM. Поэтому это сильный источник для задания 1 и более слабый кандидат на потерянную ссылку задания 2. [ARGUS, §3–4](https://arxiv.org/html/2507.15994)

## 6. Что цитировать и как сопоставлять с нашими результатами

### Для отчёта по заданию 1

Основная библиографическая связка:

- **Yambda** — происхождение данных и временной протокол.
- **DFS/Featuretools + OpenFE** — автоматический синтез и отбор признаков.
- **CAAFE** — генерация признаков LLM с обратной связью.
- **ARGUS** — нейросигнал как дополнение к сильному GBDT.
- **Perseus** — конкретный фреймворк и открытая экспериментальная конфигурация.
- **KION Challenge + eSASRec** — перенос и сильные альтернативные baseline.

Предлагаемая интерпретация ваших результатов:

> В задаче ранжирования залогированных музыкальных показов длинная история конкретной пары пользователь–трек оказывается сильным сигналом. Автоматический поиск агрегатов и последовательное представление добавляют информацию поверх ручного бейзлайна; самостоятельная последовательная модель в проверенной конфигурации не заменяет его.

Это ограниченная экспериментальная формулировка. Она не утверждает общего превосходства GBDT над трансформерами.

Для содержательного сравнения полезно показать:

1. Один candidate set, target, cutoff и набор тестовых пользователь-дней для всех моделей.
2. Ручные признаки → автопризнаки → нейросигнал → совместный вариант.
3. Результаты отдельно на известных и новых user–track парах.
4. Равный бюджет поиска гипотез для грамматики и LLM.
5. Неопределённость прироста и устойчивость по нескольким временным отрезкам.

### Для отчёта по заданию 2

В первую очередь цитировать **Personalized Popularity Awareness, RepeatNet, SAFERec, GPTRec, OneRec, RLOO, DPO, Gao et al. и Session-Level REINFORCE на Yambda**.

Ваши **0,03 → 0,17** после добавления repeat-канала разумно интерпретировать как исправление важного несоответствия модели данным. Это не показывает, что RL уже научился discovery или долгосрочному удержанию. Близость к правилу **0,18** означает, что следующим вопросом становится добавочная ценность модели сверх повторов.

Наиболее полезная сравнительная лестница:

| Сравнение | Какой вопрос отвечает |
|---|---|
| Personalized Most Popular / recency-decayed repeats | Насколько сложна задача без нейросети? |
| GPT-2 / сильный SASRec с корректным loss | Достаточен ли supervised next-item подход? |
| GPT-2 + repeat prior/gate | Устранён ли дефицит моделирования повторов? |
| Прямое RM-reranking кандидатов | Требуется ли изменение параметров политики? |
| Аналитическое \(\pi^*\) | Насколько хорошо оптимизируется одношаговая KL-цель? |
| RLOO с теми же RM, β и support | Что даёт параметрическое дообучение? |
| Несколько продолжений без RL / с feedback model | Что даёт дополнительный inference compute? |

Для новых артистов нужны два разных среза: **новый трек знакомого артиста** и **трек артиста, которого пользователь ранее не слушал**. Разделение ACARec Discovery/Exploit — подходящий образец.

### Предупреждения о несопоставимости

- **Ranking показов и retrieval каталога:** одинаковое название NDCG не делает задачи одинаковыми.
- **Следующее событие и множество событий дня:** разные targets и веса частых повторов.
- **Уникальные items и повторные события:** необходимо явно определить дедупликацию targets.
- **Frozen day-start и обновление после каждого события:** разные объёмы доступной информации.
- **Seen masking:** удаление знакомых треков радикально меняет музыкальную задачу.
- **Organic/non-organic:** разные распределения поведения; флаг не устраняет bias старой политики.
- **Full catalog, top-popular каталог и sampled negatives:** числа нельзя складывать в общий leaderboard.
- **Listen+ 50%, полный listen, 80% listen и like:** разные определения положительного.
- **GAUC и обычный AUC:** нужно указать группы, веса и обработку групп с одним классом.
- **NDCG пользователь-дня:** стоит отдельно показать длины групп и долю групп, где все объекты положительные — для них ранжирование не различает модели.
- **Окно 365 дней при 301 дне наблюдений:** в имеющихся данных это фактически вся доступная история, а не проверенный эффект годового горизонта.
- **Предсказанное продолжение сессии и retention:** первая величина не доказывает вторую.
- **Модельный rollout и реальное будущее:** оптимизация симулятора требует независимой проверки.

## 7. Список источников

Полные названия и привязки результатов приведены выше; здесь — компактный указатель первоисточников.

**Yambda и эксперименты:** [Yambda, 2505.22238](https://arxiv.org/abs/2505.22238); [данные и код](https://huggingface.co/datasets/yandex/yambda); [SEATER replication, 2512.18434](https://arxiv.org/abs/2512.18434); [CREATE, 2602.23471](https://arxiv.org/abs/2602.23471); [Position-Aware Attention, 2602.21052](https://arxiv.org/abs/2602.21052); [Variable-Length IDs, 2602.16375](https://arxiv.org/abs/2602.16375); [SID Staleness, 2604.13273](https://arxiv.org/abs/2604.13273); [ACARec, 2604.07090](https://arxiv.org/abs/2604.07090); [Ducho v2, 2409.15857](https://arxiv.org/abs/2409.15857v2); [Session-Level REINFORCE, 2607.02818](https://arxiv.org/abs/2607.02818); [MF-MDP Planning, 2607.02115](https://arxiv.org/abs/2607.02115); [Verifiers, 2609.12270](https://arxiv.org/abs/2609.12270); [CutBCE, 2610.05559](https://arxiv.org/abs/2610.05559); [Swing, 2609.16850](https://arxiv.org/abs/2609.16850); [DLRMv4](https://mlcommons.org/2026/10/dlrmv4-training/).

**KION:** [датасет, 2209.00325](https://arxiv.org/abs/2209.00325); [eSASRec, 2508.06450](https://arxiv.org/abs/2508.06450); [PNFRec, 2508.14786](https://arxiv.org/abs/2508.14786); [RecTools iALS benchmark](https://rectools.readthedocs.io/en/v0.4.1/examples/5_benchmark_iALS_with_features.html).

**Признаки и индустриальный ranking:** [DFS](https://groups.csail.mit.edu/EVO-DesignOpt/groupWebSite/uploads/Site/DSAA_DSM_2015.pdf); [Featuretools](https://featuretools.alteryx.com/en/stable/getting_started/handling_time.html); [OpenFE](https://proceedings.mlr.press/v202/zhang23ay.html); [AutoFeat, 1901.07329](https://arxiv.org/abs/1901.07329); [CAAFE, 2305.03403](https://arxiv.org/abs/2305.03403); [Constrained-Topology Agents, 2601.10820](https://arxiv.org/abs/2601.10820); [ARGUS, 2507.15994](https://arxiv.org/abs/2507.15994); [Perseus](https://github.com/perseus2026/perseus); [perseus-exps](https://github.com/perseus2026/perseus-exps).

**Alignment и RL:** [OneRec, 2502.18965](https://arxiv.org/abs/2502.18965); [OneRec Technical Report, 2506.13695](https://arxiv.org/abs/2506.13695); [OneRec-V2, 2508.20900](https://arxiv.org/abs/2508.20900); [S-DPO, 2406.09215](https://arxiv.org/abs/2406.09215); [GFlowGR, 2506.16114](https://arxiv.org/abs/2506.16114); [RLOO, 2402.14740](https://arxiv.org/abs/2402.14740); [DPO, 2305.18290](https://arxiv.org/abs/2305.18290); [Rank-GRPO, 2510.20150](https://arxiv.org/abs/2510.20150); [AdaGRPO, 2606.08480](https://arxiv.org/abs/2606.08480); [Reward Model Overoptimization](https://proceedings.mlr.press/v202/gao23h.html); [Unexpected Effectiveness of RL](https://proceedings.mlr.press/v235/silva24b.html).

**Повторы, удержание, OPE и генерация:** [RepeatNet, 1812.02646](https://arxiv.org/abs/1812.02646); [Personalized Popularity, 2409.04329](https://arxiv.org/abs/2409.04329); [Dynamics of Repeat Consumption](https://doi.org/10.1145/2566486.2568018); [Repetition and Exploration in Offline RL](https://betsyhj.github.io/publication/drl4ir_cikm2023/drl4ir_cikm2023.pdf); [SAFERec, 2412.14302](https://arxiv.org/abs/2412.14302); [RLUR, 2302.01724](https://arxiv.org/abs/2302.01724); [FeedRec, 1902.05570](https://arxiv.org/abs/1902.05570); [Bootstrapping FQE](https://proceedings.mlr.press/v139/hao21b.html); [Minimax OPE](https://proceedings.mlr.press/v119/uehara20a.html); [Top-K Off-Policy REINFORCE, 1812.02353](https://arxiv.org/abs/1812.02353); [GPTRec RL, 2403.04875](https://arxiv.org/abs/2403.04875); [Autoregressive Generation Strategies, 2409.17730](https://arxiv.org/abs/2409.17730); [GenPlugin, 2507.03568](https://arxiv.org/abs/2507.03568).

**Подтверждены библиографически, но результаты полного текста не проверены:** [SAFE feature engineering](https://doi.org/10.1007/s11704-026-60956-5); [Risk-Controlling Recommender Systems](https://doi.org/10.1145/3816736); [Analysis of Sequential Patterns, 2026](https://doi.org/10.1145/3787969); [Perseus demo](https://doi.org/10.1145/3773078.3841295).