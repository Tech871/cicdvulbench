## Launch instructions

Create .env file with following content:

```
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
HUGGINGFACE_TOKEN=
BUCKET=
CLASSIFIER=
MODEL_PATH=
DEPTH=
```

where `BUCKET` is a name of AWS bucket to use for input / output data,
`MODEL_PATH` is a path of a model on Huggingface,
make a Docker image and run the container.
