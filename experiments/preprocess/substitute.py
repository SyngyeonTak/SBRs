import random
from collections import defaultdict
from tqdm import tqdm 

def get_community_items(partitions):
    comm_items = defaultdict(list)
    for item, comm in partitions.items():
        comm_items[comm].append(item)
    return comm_items

def validate_community_partitions(partitions):
    # Create a reverse mapping from items to communities
    item_to_comm = {}
    for item, comm in partitions.items():
        if item in item_to_comm:
            print(f"Item {item} belongs to more than one community.")
            return
        item_to_comm[item] = comm

    # Get communities
    comm_items = get_community_items(partitions)

    # Print the number of communities
    num_communities = len(comm_items)
    print(f"Number of communities: {num_communities}")

    # Print the community items mapping
    print("Community items mapping:")
    # for comm, items in comm_items.items():
    #     print(f"Community {comm}: {items}")

def get_item_community(partitions, item_id):
    community = partitions.get(item_id)

    if community is not None:
        print(f"Item ID {item_id} is in community {community}.")
        return community
    else:
        print(f"Item ID {item_id} is not found in the partitions.")
        return None
    
def get_items_for_community(partitions, community_id):
    # Create a reverse mapping from community IDs to items
    community_to_items = defaultdict(list)
    for item, comm in partitions.items():
        community_to_items[comm].append(item)
    
    items = community_to_items.get(community_id, [])
    
    print('len(items): ', len(items))

    # if items:
    #     print(f"Items in community {community_id}: {items}")
    # else:
    #     print(f"No items found for community {community_id}.")
    
    return items

def augment_sessions_fixed_number_community(dataset, partitions):
    comm_items = get_community_items(partitions)
    all_items = set(item for session in dataset for item in session)  # Set of all items in the dataset
    
    # Identify items without communities
    items_with_communities = set(partitions.keys())
    no_community_items = all_items - items_with_communities
    
    augmented_sessions = []

    for session in tqdm(dataset, 'augment_sessions_fixed_number_community'):
        # Original session
        augmented_sessions.append(session[:])

        # Augmentation 1: Change a randomly selected item (excluding the last item) to another item from the same community or randomly if no community
        for _ in range(1):  # Only one augmentation
            new_session = session[:]
            if len(new_session) > 1:
                idx_to_change = random.choice(range(len(new_session) - 1))  # Exclude the last item
                item_to_replace = new_session[idx_to_change]
                comm = partitions.get(item_to_replace)
                
                if comm in comm_items and len(comm_items[comm]) > 1:
                    new_item = random.choice([i for i in comm_items[comm] if i != item_to_replace])
                else:
                    # Handle items with no community
                    new_item = random.choice(list(all_items - {item_to_replace}))
                
                new_session[idx_to_change] = new_item
            augmented_sessions.append(new_session)

        # Augmentation 2: Change the last item to another item from the same community or randomly if no community
        for _ in range(1):  # Only one augmentation
            new_session = session[:]
            last_item = new_session[-1]
            comm = partitions.get(last_item)
            
            if comm in comm_items and len(comm_items[comm]) > 1:
                new_item = random.choice([i for i in comm_items[comm] if i != last_item])
            else:
                # Handle items with no community
                new_item = random.choice(list(all_items - {last_item}))
                
            new_session[-1] = new_item
            augmented_sessions.append(new_session)

        # Augmentation 3: Change a randomly selected item (including the last) to another item from the same community or randomly if no community
        for _ in range(1):  # Only one augmentation
            new_session = session[:]
            idx_to_change = random.choice(range(len(new_session)))
            item_to_replace = new_session[idx_to_change]
            comm = partitions.get(item_to_replace)
            
            if comm in comm_items and len(comm_items[comm]) > 1:
                new_item = random.choice([i for i in comm_items[comm] if i != item_to_replace])
            else:
                # Handle items with no community
                new_item = random.choice(list(all_items - {item_to_replace}))
                
            new_session[idx_to_change] = new_item
            augmented_sessions.append(new_session)

    return augmented_sessions

def augment_sessions_length_aware_community(dataset, partitions):
    comm_items = get_community_items(partitions)
    all_items = set(item for session in dataset for item in session)  # Set of all items in the dataset
    
    # Identify items without communities
    items_with_communities = set(partitions.keys())
    no_community_items = all_items - items_with_communities
    
    augmented_sessions = []

    for session in tqdm(dataset, 'augment_sessions_length_aware_community'):
        session_length = len(session)
        
        # Original session
        augmented_sessions.append(session[:])
        
        if session_length > 1:
            # Augmentation: Change each position and the last item
            for idx in range(session_length - 1):
                new_session = session[:]
                last_item = new_session[-1]

                comm = partitions.get(last_item)

                if comm in comm_items and len(comm_items[comm]) > 1:
                    new_last_item = random.choice([i for i in comm_items[comm] if i != last_item])
                else:
                    new_last_item = random.choice(list(all_items - {last_item} | no_community_items))

                new_session[-1] = new_last_item

                # Change the item at position `idx`
                item_to_replace = new_session[idx]
                comm = partitions.get(item_to_replace)
                
                if comm in comm_items and len(comm_items[comm]) > 1:
                    new_item = random.choice([i for i in comm_items[comm] if i != item_to_replace])
                else:
                    new_item = random.choice(list(all_items - {item_to_replace} | no_community_items))
                
                new_session[idx] = new_item

                augmented_sessions.append(new_session)
                

    return augmented_sessions


def augment_sessions_length_aware_random(dataset):
    all_items = set(item for session in dataset for item in session)  # Set of all items in the dataset
    augmented_sessions = []

    for session in tqdm(dataset, 'augment_sessions_length_aware_random'):
        session_length = len(session)
        
        # Original session
        augmented_sessions.append(session[:])
        
        if session_length > 1:
            # Augmentation: Change each position and the last item
            for idx in range(session_length - 1):
                new_session = session[:]
                last_item = new_session[-1]
                new_last_item = random.choice([item for item in all_items if item != last_item])

                new_session[-1] = new_last_item

                # Change the item at position `idx`
                item_to_replace = new_session[idx]
                new_item = random.choice([item for item in all_items if item != item_to_replace])
                
                new_session[idx] = new_item

                augmented_sessions.append(new_session)
                

    return augmented_sessions