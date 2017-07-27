# pre-process input.txt

""" List of Punctuations """
# left

# for file_name in ('enron/test.txt', 'enron/train.txt', 'enron/valid.txt'):
my_string = open('enron/valid.txt', 'r').read()

my_string = my_string.replace('. ', ' . ')
my_string = my_string.replace('.\n', ' .\n')
my_string = my_string.replace(', ',' , ')
my_string = my_string.replace(',\n',' ,\n')
my_string = my_string.replace('! ', ' ! ')
my_string = my_string.replace('!\n', ' !\n')
my_string = my_string.replace('; ', ' ; ')
my_string = my_string.replace(';\n', ' ;\n')
my_string = my_string.replace('? ', ' ? ')
my_string = my_string.replace('?\n', ' ?\n')
my_string = my_string.replace(')', ' )')
my_string = my_string.replace(']', ' ]')

my_string = my_string.replace(':', ' : ')
my_string = my_string.replace('"', ' " ')
my_string = my_string.replace('\'', ' \' ')



my_string = my_string.replace('[', '[ ')
my_string = my_string.replace('(', '( ')

with open('enron/processed-valid.txt', 'w') as f:
	print(my_string, file=f)

'''
# left sided... 
# both sided...

# right sided...
/
<
>
&
%
|
=
-
{
}
#
'''