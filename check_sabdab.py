import requests
import time

AUTO_PDB_IDS = [
    "3SDY", "4FP8", "4FQY", "4NM8", "4UBD", "5K9K", "5K9Q", "5KAN", "5KAQ", "5KUY",
    "6D0U", "6NZ7", "7K37", "7K39", "7K3A", "7K3B", "7RDH", "7X6L", "8Y7O", "8YVN"
]

SABDAB_API = "https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/api/sabdab"

def query_sabdab(pdb_id):
    url = f"{SABDAB_API}/{pdb_id.lower()}/"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            return None
    except Exception as e:
        print(f"  Error: {e}")
        return None

def main():
    print("Checking AUTO_PDB_IDS via SAbDab API...\n")
    for pdb_id in AUTO_PDB_IDS:
        print(f"{pdb_id}:")
        data = query_sabdab(pdb_id)
        if data is None:
            print("  No data from SAbDab (or error)")
            continue

        antibody_chains = set()
        antigen_chains = set()
        if 'chains' in data:
            for ch_id, ch_info in data['chains'].items():
                if ch_info.get('type') == 'antibody':
                    antibody_chains.add(ch_id)
                elif ch_info.get('type') == 'antigen':
                    antigen_chains.add(ch_id)

        if antibody_chains:
            print(f"  Antibody chains: {', '.join(sorted(antibody_chains))}")
        else:
            print("  No antibody chains found in SAbDab")

        if antigen_chains:
            print(f"  Antigen chains: {', '.join(sorted(antigen_chains))}")
        else:
            print("  No antigen chains found in SAbDab")

        # Показать тип каждой цепи
        if 'chains' in data:
            chain_types = {ch: info.get('type', 'unknown') for ch, info in data['chains'].items()}
            print(f"  All chains: {chain_types}")

        print()
        time.sleep(0.5)

if __name__ == "__main__":
    main()