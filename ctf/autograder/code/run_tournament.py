import os
import yaml
import glob
import itertools
import capture
import csv
from datetime import datetime
    
from contextlib import contextmanager
import sys
import multiprocessing as mp
from multiprocessing import Manager
import argparse

@contextmanager
def suppress_stdout_and_stderr(path=os.devnull):
    with open(path, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def run_two_team(team1, team2, num_repeats, log_path=os.devnull):
    layouts = [
        'defaultCapture',
        'fastCapture',
        'alleyCapture',
        'mediumCapture',
        'distantCapture',
        'strategicCapture',
    ]
    team1_wins = 0
    
    with suppress_stdout_and_stderr(log_path):
        for _ in range(num_repeats):
            for l in layouts:
                pacman_cmd = f'python capture.py -r ./submissions/{team1}.py -b ./submissions/{team2}.py -l {l} -c -q'
                args = capture.readCommand(pacman_cmd.split()[2:])
                games = capture.runGames(**args)
                # Take the average of the game scores. Note that there should be
                # only one game in games, unless `-n` is used in pacman.py
                scores = [game.state.data.score for game in games]
                game_score = sum(scores) / len(scores)
                if game_score > 0:
                    team1_wins += 1
                elif game_score < 0:
                    team1_wins -= 1
                
    if team1_wins > 0:
        team1_score = 3
        team2_score = 0
    elif team1_wins < 0:
        team1_score = 0
        team2_score = 3
    else:
        team1_score = 1
        team2_score = 1
    
    team1_name = sub_name_to_names(team1)
    team2_name = sub_name_to_names(team2)
    score_board[team1_name] = score_board.get(team1_name, 0) + team1_score
    score_board[team2_name] = score_board.get(team2_name, 0) + team2_score
    team1_results = match_board.get(team1_name, {})
    team1_results[team2_name] = team1_score
    match_board[team1_name] = team1_results
    team2_results = match_board.get(team2_name, {})
    team2_results[team1_name] = team2_score
    match_board[team2_name] = team2_results
    return f'{team1_name} vs {team2_name}: {team1_score}'

def sub_name_to_names(sub_name):
    submitters = []
    for name in meta[sub_name][':submitters']:
        submitters.append(name[':name'])
    return ','.join(submitters)
    
def main(args):
    global score_board
    global match_board
    global meta
    
    submission_path = args.submission_path
    meta_path = os.path.join(submission_path, 'submission_metadata.yml')
    with open(meta_path, 'r') as f:
        meta = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.name_list, 'r') as f:
        name_list = [line.strip() for line in f.readlines()]
        
    qualified_submissions = []
    for sub_name, v in meta.items():
        if v[':submitters'][0][':name'] == 'Zhenyu Jiang':
            continue
        if sub_name_to_names(sub_name) not in name_list:
            continue
        for item in v[':results']['leaderboard']:
            if item['name'] == 'Winning Rate vs. baselineTeam':
                wining_rate = item['value']
                break
        if wining_rate >= 0.5:
            qualified_submissions.append(sub_name)
    if args.num_teams > 0:
        qualified_submissions = qualified_submissions[:args.num_teams]
    print(f'{len(qualified_submissions)} teams qualified. {len(qualified_submissions) * (len(qualified_submissions) - 1) / 2} games to be played.')


    date_str = datetime.now().strftime('%Y-%m-%d')
    dst_dir = './submissions'
    log_dir = f'./logs/logs-{date_str}'
    # clean up
    if os.path.exists(dst_dir):
        os.system(f'rm -r {dst_dir}')
    if os.path.exists(log_dir):
        os.system(f'rm -r {log_dir}')
    os.makedirs(dst_dir)
    os.makedirs(log_dir)
    os.makedirs('results', exist_ok=True)
    for sub_name in qualified_submissions:
        src_path = f'{submission_path}/{sub_name}/myTeam.py'
        dst_path = f'{dst_dir}/{sub_name}.py'
        os.system(f'cp {src_path} {dst_path}')

    # run tournament
    if args.num_processes > 0:
        with Manager() as manager:
            score_board = manager.dict()
            match_board = manager.dict()
            pool = mp.Pool(processes=args.num_processes)
            for team1, team2 in itertools.combinations(qualified_submissions, 2):
                team1_name = sub_name_to_names(team1)
                team2_name = sub_name_to_names(team2)
                log_path = f'./{log_dir}/{team1_name} vs {team2_name}.log'
                pool.apply_async(run_two_team, args=(team1, team2, args.num_repeats, log_path), callback=lambda x: print(x))
            pool.close()
            pool.join()
            score_board = dict(score_board)
            match_board = {k: dict(v) for k, v in dict(match_board).items()}
    
    else:
        score_board = {}
        match_board = {}
        for team1, team2 in itertools.combinations(qualified_submissions, 2):
            team1_name = sub_name_to_names(team1)
            team2_name = sub_name_to_names(team2)
            log_path = f'./{log_dir}/{team1_name} vs {team2_name}.log'
            print(run_two_team(team1, team2, args.num_repeats, log_path=log_path))

    print('Score Board:') 
    for k, v in score_board.items():
        print(f'{k}: {v}')
    # turn into csv
    with open(f'results/score_board-{date_str}.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['team', 'score'])
        for k, v in score_board.items():
            writer.writerow([k, v])
            
    with open(f'results/match_board-{date_str}.csv', 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow([''] + [sub_name_to_names(x) for x in qualified_submissions])
        for k, v in match_board.items():
            row = [k]
            for team in qualified_submissions:
                row.append(v.get(sub_name_to_names(team), ''))
            writer.writerow(row)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--submission-path', type=str, required=True)
    parser.add_argument('-l', '--name-list', type=str, required=True)
    parser.add_argument('-n', '--num-processes', type=int, default=1)
    parser.add_argument('-r', '--num-repeats', type=int, default=3)
    parser.add_argument('--num-teams', type=int, default=-1, help='-1 means all')
    args = parser.parse_args()
    main(args)