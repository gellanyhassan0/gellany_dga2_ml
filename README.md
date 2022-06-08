# gellany_dga2_ml

we forked and refactory from https://github.com/SuperCowPowers/data_hacking/tree/master/dga_detection and combine with https://github.com/gellanyhassan0/gellany_dga

<code>python3 gellany_dga2_ml.py -fn alexa_100k.csv -fd dga_domains.txt</code><br>
<code>python3 gellany_dga2_ml.py -d isqekc</code><br>
threshold 0.018782003473122023</code><br>
math.exp(log_prob / (transition_ct or 1) 0.0034698353681716747</code><br>
Domain isqekc is DGA!</code><br>

<code>python3 gellany_dga2_ml.py -d google</code><br>
threshold 0.018782003473122023</code><br>
math.exp(log_prob / (transition_ct or 1) 0.03488552714083014</code><br>
Domain google is clean!</code><br>

<code>python3 gellany_dga2_ml.py -d congresomundialjjrperu2009
threshold 0.018782003473122023
math.exp(log_prob / (transition_ct or 1) 0.03713399431330488
Domain congresomundialjjrperu2009 is clean!</code><br>

<code>python3 gellany_dga2_ml.py -h
usage: gellany_dga2_ml.py [-h] [-d DOMAIN] [-fn FILE_NORMAL] [-fd FILE_DGA]

optional arguments:
  -h, --help            show this help message and exit
  -d DOMAIN, --domain DOMAIN
                        Domain to check
  -fn FILE_NORMAL, --file_normal FILE_NORMAL
                        File with normal. One per line
  -fd FILE_DGA, --file_dga FILE_DGA
                        File with dga. One per line</code><br>

