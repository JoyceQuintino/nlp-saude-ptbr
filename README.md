# Classificação de textos de saúde em português

Comparação de três abordagens para classificar perguntas de pacientes em PT-BR
(tarefa: prever a especialidade médica de quem respondeu, 10 classes):

| Modelo | Stack | F1 macro (teste) |
|---|---|---|
| TF-IDF + Regressão Logística (baseline) | scikit-learn | 0,782 |
| BiLSTM | TensorFlow/Keras | 0,781 |
| BERTimbau (fine-tuning) | PyTorch + Hugging Face | **0,845** |

Inclui rastreamento de experimentos (MLflow), API (FastAPI), Docker e CI (GitHub Actions).

> Uso educacional e de portfólio. O modelo **não** substitui avaliação médica.

## Dados
Dataset: **MedPT** (`AKCIT/MedPT`, Hugging Face): ~384 mil pares pergunta-resposta de
pacientes e médicos em português do Brasil, com especialidade, condição e tipo de pergunta.
Artigo: Färber et al., *MedPT: A Massive Medical Question Answering Dataset for
Brazilian-Portuguese Speakers* (LREC 2026).

- **Licença e uso dos dados:** os dados pertencem aos autores do MedPT e seguem a licença
  indicada no [dataset card](https://huggingface.co/datasets/AKCIT/MedPT). Este repositório
  **não redistribui** os dados: eles são baixados do Hugging Face por quem executar o código,
  sob os termos da licença original. Os créditos do dataset são dos autores (Färber et al.).
- Os dados não são versionados neste repositório. Para gerá-los:

```bash
python -m src.prepare_medpt --task specialty --top 10 --max_per_class 2000
```

Decisões de preparação (importantes para não inflar a métrica):
- O dataset tem ~2 respostas por pergunta. Mantemos **uma linha por pergunta** e descartamos
  perguntas com rótulos conflitantes, para não haver o mesmo texto em treino e teste.
- `specialty` é a especialidade de **quem respondeu**, não necessariamente da pergunta. Rótulos com
  várias especialidades (ex.: "Psicólogo, Psicanalista") são descartados, pois se sobrepõem às classes simples.
- `question_type` foi anotado por um LLM no próprio dataset (rótulo sintético).
- Limitamos a `max_per_class` exemplos por classe para o treino caber numa GPU gratuita.

## Como rodar
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m src.prepare_medpt --task specialty --top 10 --max_per_class 2000
python -m src.baseline_tfidf
python -m src.train_keras
python -m src.tune_bert --epochs 2      # busca de hiperparâmetros (escolhe pela validação)
python -m src.train_bert --lr <melhor_lr> --batch_size <melhor_bs> --epochs 3   # modelo final

mlflow ui                               # compara os experimentos
pytest -q
```

### No Google Colab (GPU gratuita)
Runtime > Change runtime type > T4 GPU. O Colab já traz PyTorch, TensorFlow e Transformers;
instale só o que falta:
```python
!git clone https://github.com/JoyceQuintino/nlp-saude-ptbr.git
%cd nlp-saude-ptbr
!pip install -q datasets mlflow
!python -m src.prepare_medpt --task specialty --top 10 --max_per_class 2000
!python -m src.baseline_tfidf
!python -m src.train_keras
!python -m src.tune_bert --epochs 2
!python -m src.train_bert --lr 5e-5 --batch_size 16 --epochs 3
```
Baixe `models/bert/`, `reports/` e `mlruns/` ao final (ou envie o modelo ao Hugging Face Hub).

## API
```bash
uvicorn api.main:app --reload
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"text": "Estou com dor de cabeça forte há dias, que especialista procuro?"}'

docker build -t nlp-saude . && docker run -p 8000:8000 nlp-saude
```
O modelo treinado não é versionado no Git (`models/` está no `.gitignore`): o `docker build`
funciona localmente depois do treino. Para deploy público, publique o modelo no Hugging Face Hub.

## Resultados e análise de erros
Conjunto de teste: 3.000 perguntas (300 por especialidade), a mesma divisão para os três
modelos (70/15/15, semente 42). Base: 10 especialidades x 2.000 perguntas.

- **Ajuste de hiperparâmetros (BERT):** busca em grade de taxa de aprendizado (1e-5, 2e-5, 5e-5)
  com lote 16, escolhida pelo F1 macro de **validação**. Melhor: 5e-5 (F1 macro de validação 0,829).
- **BERTimbau superou os dois baselines em cerca de 6 pontos de F1 macro** (0,845 contra 0,782 e 0,781).
- **Por classe:** melhores Oftalmologista (0,96) e Ortopedista/traumatologista (0,93);
  piores Psicólogo (0,71), Psiquiatra (0,74) e Neurologista (0,77).
- Resultados de uma única execução (uma semente), sem intervalo de confiança; o grid testou
  poucos valores e o melhor ficou na borda (5e-5), então há espaço para ajuste.

### Análise de erros
O BERTimbau errou 464 das 3.000 perguntas de teste (15,5%). As 12 confusões mais comuns
somam 240 erros (52% do total):

| Real → previsto | Erros |
|---|---|
| Psicólogo → Psiquiatra | 61 |
| Psiquiatra → Psicólogo | 48 |
| Urologista → Ginecologista | 22 |
| Neurologista → Psicólogo | 20 |
| Ginecologista → Urologista | 16 |
| Psiquiatra → Neurologista | 14 |
| Neurologista → Dermatologista | 11 |
| Neurologista → Otorrino | 10 |
| Psicólogo → Neurologista | 10 |
| Otorrino → Neurologista | 10 |
| Dentista → Otorrino | 9 |
| Neurologista → Dentista | 9 |

Padrões observados (as explicações abaixo são hipóteses, não foram testadas):
- **Psicólogo × Psiquiatra** (109 erros, 23% do total, nos dois sentidos): os temas se
  sobrepõem (ansiedade, depressão, antidepressivos). Ex.: uma pergunta sobre depressão pós-parto
  com antidepressivos, respondida por psicólogo, foi prevista como psiquiatra.
- **Urologista × Ginecologista** (38 erros): perguntas de saúde sexual e infecções genitais
  servem às duas especialidades.
- **Neurologista** aparece em 84 erros (18%), confundido com psicólogo, psiquiatra, otorrino,
  dermatologista e dentista, pois sintomas como dor, tontura e formigamento são comuns a várias áreas.
- **Rótulo = quem respondeu, não o assunto.** Algumas perguntas poderiam ser legitimamente
  respondidas por outra especialidade (ex.: dúvida sobre suplemento para distrofia muscular,
  respondida por neurologista e prevista como ortopedista), então parte dos "erros" é ambiguidade do rótulo.
- **Textos curtos erram um pouco mais:** tamanho médio de 157 caracteres nos erros contra
  171 nos acertos, diferença pequena.

## Licença
O código deste repositório está sob a licença [MIT](LICENSE). A licença do código **não se
aplica** ao dataset MedPT nem ao modelo BERTimbau (`neuralmind/bert-base-portuguese-cased`),
que mantêm as licenças de seus autores.

## Decisões e limitações
- F1 macro como métrica principal por causa do desbalanceamento de classes.
- Ajuste de hiperparâmetros (taxa de aprendizado, com lote 16) por busca em grade,
  escolhendo pelo F1 macro de **validação**; o teste só é usado uma vez, no modelo final.
- Pesos por classe na loss; melhor checkpoint escolhido na validação, nunca no teste.
- **Rótulo indireto:** o alvo é a especialidade de quem respondeu, não um diagnóstico nem a
  especialidade "correta" para a pergunta. Isso limita o teto de desempenho (ver análise de erros).
- **Subconjunto:** 10 especialidades de rótulo único, 2.000 perguntas por classe, para caber em
  GPU gratuita. Os resultados não se estendem ao dataset completo nem a outras especialidades.
- **Origem dos dados:** perguntas de pacientes em plataforma online, com linguagem informal,
  erros de digitação e possível viés de quem usa esse tipo de serviço.
- **Avaliação:** uma única execução (uma semente), sem intervalo de confiança; a diferença de
  cerca de 6 pontos sobre os baselines é consistente, mas diferenças menores não seriam conclusivas.
- **Risco em saúde:** o modelo não é validado clinicamente e não deve ser usado para triagem
  ou para orientar pacientes. Uso restrito a estudo e portfólio.