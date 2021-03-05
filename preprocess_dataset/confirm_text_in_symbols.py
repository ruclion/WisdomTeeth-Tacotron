_pad = '_'
_eos = '~'
_tone = '123456'
_letters = 'abcdefghijklmnopqrstuvwxyz'
_space = ' '


# Export all symbols:
symbols = [_pad] + [_eos] + list(_tone) + list(_letters) + [_space]


out_txt_path = ['training_data/train.txt', 'training_data/val.txt', 'training_data/test.txt']

for path in out_txt_path:
    f = open(path, encoding='utf-8')
    out_list = []
    for x in f.readlines():
        x = x.strip()
        text = x.split('|')[5]
        tag = True
        for i in text:
            if (i in symbols) is False:
                tag = False
                break
        if tag:
            out_list.append(x)
    
    f.close()
    f = open(path, 'w', encoding='utf-8')
    for x in out_list:
        f.write(x + '\n')


