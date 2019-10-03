import numpy as np
import pandas as pd
import pandas.tseries.offsets as offsets
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true')
args = parser.parse_args()


class Util:

    @classmethod
    def iif(cls, cond, iftrue, ifelse):
        if cond:
            return iftrue
        else:
            return ifelse


class Generator:

    def __init__(self):
        pass

    def choice_prob(self, prob, iftrue, ifelse):
        if self.r.rand() < prob:
            return iftrue
        else:
            return ifelse

    def run_all(self, seed=71, n=100):
        self.r = np.random.RandomState(seed)
        ds = []
        for i in range(n):
            print(i)
            data, scores, target = self.run()
            ds.append(data + scores + target)

        columns = self.column_names()
        df = pd.DataFrame(ds, columns=columns)
        return df

    def run(self):

        # 被保険者の情報
        # 年齢・体重・BMIに応じてスコアが設定されるとする
        insured_age = self.r.choice(range(5, 80))
        insured_sex = self.r.choice(['Male', 'Female'], p=[0.6, 0.4])
        height = 160.0 + Util.iif(insured_sex == 'Male', 10.0, 0.0) + self.r.randn() * 8
        bmi = 22.0 + self.r.randn() * 3.0
        weight = height * height * bmi / 10000.0

        if insured_age <= 15:
            score_01 = 0.1
        elif 60 <= insured_age < 70:
            score_01 = 0.2
        elif 70 <= insured_age:
            score_01 = 0.3
        else:
            score_01 = 0.0

        if weight >= 85.0:
            score_02 = 0.2
        else:
            score_02 = 0.0

        if bmi <= 19.0:
            score_03 = (19.0 - bmi) * 0.15
        elif bmi >= 25.0:
            score_03 = (bmi - 25.0) * 0.15
        else:
            score_03 = 0.0

        # 商品情報
        # A1-A3, B1-B3, C1-C3, D1, E1
        # D, E, 2の時にスコアを設定している

        product = self.r.choice(list('ABCDE'), p=[0.5, 0.1, 0.25, 0.1, 0.05])
        is_prodtype_1 = product in list('ABD')
        is_prodtype_2 = product in list('CE')
        is_prodtype_a = product in list('ABC')
        is_prodtype_b = product in list('DE')

        if is_prodtype_a:
            product_sub = self.r.choice([1, 2, 3], p=[0.4, 0.2, 0.4])
        else:
            product_sub = 1
        product = '{}{}'.format(product, product_sub)

        if is_prodtype_b:
            score_21 = 0.2
        else:
            score_21 = 0.0
        if product_sub == 2:
            score_22 = 0.1
        else:
            score_22 = 0.0

        # 保険金額 - 商品タイプに応じて基本となる金額が異なる
        # 保険金額が大きいとスコアが設定される
        amount_raw = self.r.choice(range(1, 11))
        if is_prodtype_1:
            amount = amount_raw * 1000  * 1000
        else:
            amount = amount_raw  * 1000

        if amount_raw > 5:
            score_23 = 0.1
        else:
            score_23 = 0.0

        # 申込日もスコアなし
        date_start = pd.to_datetime('2015/1/1')
        date_end = pd.to_datetime('2016/12/31')
        days = (date_end - date_start).days

        app_date = date_start + offsets.Day(self.r.choice(range(days)))
        app_year, app_month, app_day = app_date.year, app_date.month, app_date.day
        application_date = '{}/{}/{}'.format(app_year, app_month, app_day)

        # 医療情報
        # a1, a1とa2の差, a3が5以上でスコアが設定
        # bはスコアが無い
        medical_info_a1 = int(250.0 + 100.0 * self.r.rand() + 100.0 * self.r.randn())
        medical_info_a2 = int(200.0 + 100.0 * self.r.rand() + 100.0 * self.r.randn())
        medical_info_a3 = self.r.poisson(lam=2)

        medical_info_b1 = int(10.0 + 10.0 * self.r.rand())
        medical_info_b2 = self.r.choice([1,2,3,9], p=[0.5, 0.25, 0.2, 0.05])
        medical_info_b3 = self.r.choice(list('ABCDEFGH') + list('abcde') + ['1', '2', '3', '4'])

        medical_info_c1 = self.r.choice([np.nan, self.r.poisson(lam=1.5)], p=[0.3, 0.7])
        medical_info_c2 = self.r.choice([np.nan, self.r.uniform(8.0, 22.0)], p=[0.8, 0.2])

        if medical_info_a1 > 350:
            score_41 = 0.2
        else:
            score_41 = 0.0

        medical_info_a_diff = medical_info_a1 - medical_info_a2
        score_42 = max(np.abs(medical_info_a_diff) - 100.0, 0.0) / 800.0

        if medical_info_a3 >= 5:
            score_43 = 0.2
        else:
            score_43 = 0.0

        if np.isnan(medical_info_c1):
            score_44 = -0.1
        elif medical_info_c1 >= 3:
            score_44 = 0.1
        else:
            score_44 = 0.0

        # 医療情報のバイナリ値
        # 1-5番目はスコアと関係がある。
        # 6-7番目は女性のみスコアと関係がある
        # 8-10番目はスコアと関係が無い
        medical_keyword_probs = np.array([
            0.8, 0.5, 0.2, 0.05, 0.02, 0.4, 0.1, 0.8, 0.3, 0.05,
        ])
        medical_keywords = []
        for prob in medical_keyword_probs:
            medical_keywords.append(self.r.choice([0, 1], p=[1 - prob, prob]))

        mkeys = medical_keywords[:5]
        mprobs = medical_keyword_probs[:5]
        mkeys_sum = np.array(mkeys).sum()
        mkeys_score = 1.0 / mprobs * 0.01

        score_51 = np.sum(np.array(mkeys) * mkeys_score)
        if mkeys_sum >= 4:
            score_52 = 0.5
        elif mkeys_sum >= 3:
            score_52 = 0.3
        elif mkeys_sum >= 2:
            score_52 = 0.1
        else:
            score_52 = 0.0

        score_53 = 0.0
        if insured_sex == 'Female':
            if medical_keywords[5] == 1 or medical_keywords[6] == 1:
                score_53 += 0.1

        # noise
        score_noise = self.r.uniform(-0.2, 0.2)

        data = [insured_age, insured_sex, height, weight,
                product, amount, application_date,
                medical_info_a1, medical_info_a2, medical_info_a3,
                medical_info_b1, medical_info_b2, medical_info_b3,
                medical_info_c1, medical_info_c2] + medical_keywords

        score_elements = [score_01, score_02, score_03, score_21, score_22, score_23,
                          score_41, score_42, score_43, score_44, score_51, score_52, score_53, score_noise]
        score = np.array(score_elements).sum()
        scores = [score] + score_elements

        target = [Util.iif(score >= 0.8, 1, 0)]

        return data, scores, target

    def column_names(self):
        return self.column_names_data() + self.column_names_scores() + self.column_names_target()

    def column_names_data(self):
        data = (['age', 'sex', 'height', 'weight',
                 'product', 'amount', 'date',
                 'medical_info_a1', 'medical_info_a2', 'medical_info_a3',
                 'medical_info_b1', 'medical_info_b2', 'medical_info_b3',
                 'medical_info_c1', 'medical_info_c2']
                + ['medical_keyword_{}'.format(i) for i in range(1, 11)])
        return data

    def column_names_scores(self):
        score_elements = ['score_01', 'score_02', 'score_03', 'score_21', 'score_22', 'score_23',
                          'score_41', 'score_42', 'score_43', 'score_44',
                          'score_51', 'score_52', 'score_53', 'score_noise']
        scores = ['score'] + score_elements
        return scores

    def column_names_target(self):
        target = ['target']
        return target


if __name__ == '__main__':
    gen = Generator()
    if args.test:
        n_tr = 100
    else:
        n_tr = 10000

    df = gen.run_all(n=n_tr * 2)
    cols_data = gen.column_names_data()
    cols_target = gen.column_names_target()

    # df[:n_tr].to_csv('train_debug.csv', index=False, sep='\t')
    # df[n_tr:].to_csv('test_debug.csv', index=False, sep='\t')
    df[:n_tr][cols_data + cols_target].to_csv('train.csv', index=False, sep=',')
    df[n_tr:][cols_data].to_csv('test.csv', index=False, sep=',')
