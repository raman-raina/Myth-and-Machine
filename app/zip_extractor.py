import zipfile


def zip(d1, d2):
	zipfile.ZipFile(d1, 'r').extractall('dataset')
	zipfile.ZipFile(d2, 'r').extractall('new')