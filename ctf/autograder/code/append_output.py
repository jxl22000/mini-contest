import json


if __name__ == '__main__':
    try:
        with open('/autograder/results/results.json', 'r') as output_file:
            score_fields = json.load(output_file)
    except Exception:
        score_fields = {'output': ''}

    try:
        with open('/autograder/results/output.txt', 'r') as f:
            output = f.read()
    except Exception:
        output = ''

    score_fields['output'] += output
    with open('/autograder/results/results.json', 'w') as output_file:
        json.dump(score_fields, output_file)
