Project Graph2Seq IGND with 3 different graph construction methods. This is bachelor thesis in HUST.

The model source code is forked from [IGND, Fei et al., 2021](https://github.com/sion-zcfei/ignd), the preprocess code of graph construction is in 2 preprocess_* files and annotation folder.

## Preprocess & Annotate data
First download SQuAD v1.1 from [SQuAD-explorer](https://github.com/rajpurkar/SQuAD-explorer), then process it.
1. cd to src/ folder and run preprocess file. File `preprocess_stanza.py` use for annotate head and boundary construction, and file `preprocess_allennlp_spacy.py` use for annotate coref method.
```
python preprocess_stanza.py -i <path_to_raw_squad_data> -o <path_to_output>
```
```
python preprocess_allennlp_spacy.py -i <path_to_raw_squad_data> -o <path_to_output>
```



## Training and test
### Training
1. Download vocal model 70k. https://fastupload.io/5b4qrZISna5fujd/file.
2. Modify config file in src/config with your right data and vocab path. Change `trainset`, `devset`, `testset`, `saved_vocab_file` option to your path, `pretrained` option is empty if there is no saved checkpoint else it equals to `out_dir` option.
3. Set `only_test` to False.
4. Run `main.py` to training model.
```
python main.py -config <path_to_config>
```

***Note***: if you saved checkpoint and do continuous training, go to `core/utils/logger.py` and comment some lines in `class DummyLogger` and add your params.saved path to option `pretrained` in yml config file.
```
def __init__(self, config, dirname=None, pretrained=None):
    self.config = config
    if dirname is None:
        if pretrained is None:
            raise Exception('Either --dir or --pretrained needs to be specified.')
        self.dirname = pretrained
    else:
        self.dirname = dirname
        # if os.path.exists(dirname):
        #     raise Exception('Directory already exists: {}'.format(dirname))
        # os.makedirs(dirname)
        # os.mkdir(os.path.join(dirname, 'metrics'))
        self.log_json(config, os.path.join(self.dirname, Constants._CONFIG_FILE))
    if config['logging']:
        self.f_metric = open(os.path.join(self.dirname, 'metrics', 'metrics.log'), 'a')
```

### Test
Change option `only_test` to True then run above command ```python main.py ...``` 
