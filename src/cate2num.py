import pandas as pd
from tqdm import tqdm
from sklearn.utils import shuffle
from tqdm import tqdm


df = pd.read_csv('./beijing.csv', encoding='gbk')
label2idx = {'EPN': 0, 'DCN': 1, 'CN': 2, 'HBN': 3}
df.scene = df.scene.map(label2idx)

df.fillna(-1, inplace=True)
df = shuffle(df, random_state=2020)

unselected = ["whois_blk_regist_date", "base_blk_id", "num_minip", "num_maxip"]
to = df.columns.values.tolist()
for i in to:
    if i in unselected:
        df.drop([i], axis=1, inplace=True)
to = sc.columns.values.tolist()


categorical = [38, 39]
numerical = [i for i in range(40) if i not in [0, 38, 39]]
rescale = ['whois_blk_ip_num']


def cate2num(df):
    cate = {}
    cateidx = {}
    flag = 0
    index = [1] * len(categorical)
    col = 0
    for j in categorical:
        for i in tqdm(range(df.shape[0])):
            if df.iloc[i, j] not in list(cate.keys()):
                # [1, 0] 1 is the index for those whose appear times <= 10   0 indicates the appear times
                cate[df.iloc[i, j]] = [1, 0]
            cate[df.iloc[i, j]][1] += 1
            if cate[df.iloc[i, j]][0] == 1 and cate[df.iloc[i, j]][1] >= 10:
                index[col] += 1
                cate[df.iloc[i, j]][0] = index[col]
                # cateidx[df.iloc[i, j]] = index[col]
        col += 1
        # if col < len(categorical):
        #     index[col] += index[col-1]
        # df.iloc[:, j] = df.iloc[:, j].map(cateidx)
    offset = []
    kinds = 0
    for m in range(len(index)):
        offset.append(kinds)
        kinds += index[m]
    col = 0
    for j in categorical:
        for i in tqdm(range(df.shape[0])):
            df.iloc[i, j] = cate[df.iloc[i, j]][0] + offset[col]
        col += 1

    return df


def regularit(df, col=None):
    if col is None:
        columns = df.columns.tolist()
    else:
        columns = col

    for c in columns:
        d = df[c]
        MAX = d.max()
        MIN = d.min()
        df[c] = ((d - MIN) / (MAX - MIN)).tolist()
    return df


df = cate2num(df)
df = regularit(df, rescale)
df.to_csv('beijing_cate2id.csv', index=False)
