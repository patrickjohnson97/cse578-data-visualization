import math
import os
import shutil

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

current_directory = os.path.dirname(os.path.realpath(__file__))

output_directory = os.path.join(current_directory, 'output/')

if os.path.isdir(output_directory):
	shutil.rmtree(output_directory)

os.makedirs(output_directory)

def _get_data():
	data_directory    = os.path.join(current_directory, os.pardir, os.pardir, 'data/')
	adult_data_path   = os.path.join(data_directory, 'adult.data')

	return np.genfromtxt(adult_data_path, delimiter=', ', unpack=True, dtype=np.object)

def _make_friendly_name(title):
	file_name = title.replace('<=', 'lt').replace('>', 'gt')
	file_name = ''.join(c for c in file_name if c.isalnum() or c == ' ')

	return file_name

def _is_normal(data, alpha=1e-3):
	_, p = stats.normaltest(data)

	return p > alpha

def _make_histogram(data, title, x_label, y_label, log=False):
	output_file_path = os.path.join(output_directory, '%s.png' % _make_friendly_name(title))

	n = len(data)

	if _is_normal(data):
		k = int(math.sqrt(n))
	else:
		# use Sturge's Formula
		k = math.ceil(math.log(n) + 1)

	if '>' in title:
		color = '#33E291'
	else:
		color = None

	plt.style.use('ggplot')
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.hist(data, bins=k, log=log, color=color)
	plt.tight_layout()
	plt.savefig(output_file_path, dpi=400)
	plt.close()

def _make_bar_chart(x_labels, y_values, title, x_label, y_label):
	output_file_path = os.path.join(output_directory, '%s.png' % _make_friendly_name(title))

	if '>' in title:
		color = '#33E291'
	else:
		color = None

	x = range(len(x_labels))

	plt.style.use('ggplot')
	plt.title(title)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.barh(x, y_values, color=color)
	plt.yticks(x, x_labels)
	plt.tight_layout()
	plt.savefig(output_file_path, dpi=400)
	plt.close()

adult_data = _get_data()


# create histograms for top continuous features
age           = adult_data[0,  :].astype(np.int)
fnlwgt        = adult_data[2,  :].astype(np.int)
education_num = adult_data[4,  :].astype(np.int)
capital_gain  = adult_data[10, :].astype(np.int)
income        = adult_data[14, :]

income_lt_50k = income == b'<=50K'
income_gt_50k = income == b'>50K'

# fnlwgt
fnlwgt_lt_50k = fnlwgt[income_lt_50k]
fnlwgt_gt_50k = fnlwgt[income_gt_50k]

_make_histogram(fnlwgt_lt_50k, 'fnlwgt for Income <= 50K', 'fnlwgt', 'Number of people')
_make_histogram(fnlwgt_gt_50k, 'fnlwgt for Income > 50K',  'fnlwgt', 'Number of people')

# age
age_lt_50k = age[income_lt_50k]
age_gt_50k = age[income_gt_50k]

_make_histogram(age_lt_50k, 'Age for Income <= 50K', 'Age', 'Number of people')
_make_histogram(age_gt_50k, 'Age for Income > 50K',  'Age', 'Number of people')

# capital-gain
capital_gain_lt_50k = capital_gain[income_lt_50k]
capital_gain_gt_50k = capital_gain[income_gt_50k]

_make_histogram(capital_gain_lt_50k, 'Capital Gain for Income <= 50K', 'Capital Gain', 'Number of people', log=True)
_make_histogram(capital_gain_gt_50k, 'Capital Gain for Income > 50K',  'Capital Gain', 'Number of people', log=True)

# education-num
education_num_lt_50k = education_num[income_lt_50k]
education_num_gt_50k = education_num[income_gt_50k]

_make_histogram(education_num_lt_50k, 'Education Num for Income <= 50K', 'Education Num', 'Number of people')
_make_histogram(education_num_gt_50k, 'Education Num for Income > 50K',  'Education Num', 'Number of people')


# create bar chart for top categorical features
relationship = adult_data[7, :].astype(np.unicode_)

x_labels = sorted(np.unique(relationship))
x_labels = [str(x_label) for x_label in x_labels]

y_values = []

for x_label in x_labels:
	count = len(relationship[(relationship == x_label) & income_lt_50k])

	y_values.append(count)

_make_bar_chart(x_labels, y_values, 'Relationship for Income <= 50K', 'Number of People', 'Relationship')

y_values = []

for x_label in x_labels:
	count = len(relationship[(relationship == x_label) & income_gt_50k])

	y_values.append(count)

_make_bar_chart(x_labels, y_values, 'Relationship for Income > 50K', 'Number of People', 'Relationship')
