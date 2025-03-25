import pandas as pd
import os

def get_ordered_actions(csv_files):
    # Dictionary to store action_id -> action_name mappings
    action_map = {}
    
    # Process each CSV file
    for file in csv_files:
        if not os.path.exists(file):
            print(f"Warning: {file} not found")
            continue
            
        df = pd.read_csv(file)
        
        # Add all action_id -> action_name mappings to our dictionary
        action_pairs = df[['action_id', 'action_name']].drop_duplicates()
        for _, row in action_pairs.iterrows():
            action_map[row['action_id']] = row['action_name']
    
    # Sort by action_id and create the final list
    ordered_actions = [action_map[i] for i in sorted(action_map.keys())]
    return ordered_actions

def main():
    base_path = "/home/milkyway/Desktop/Student Thesis/Datasets/RGB_frames"
    csv_files = [
        os.path.join(base_path, "train_orginal.csv"),
        os.path.join(base_path, "val_orginal.csv"),
        os.path.join(base_path, "test.csv")
    ]
    
    action_list = get_ordered_actions(csv_files)
    
    # Print the results
    print("\nOrdered Action List:")
    for i, action in enumerate(action_list):
        print(f"{i}: {action}")
    
    # Optionally save to file
    output_file = os.path.join(base_path, "action_list.txt")
    with open(output_file, 'w') as f:
        for i, action in enumerate(action_list):
            f.write(f"{i}: {action}\n")
    
    print(f"\nAction list saved to: {output_file}")

if __name__ == "__main__":
    main()