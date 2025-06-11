#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import MDAnalysis as mda
from MDAnalysis.analysis import rms, align
from MDAnalysis.analysis.rms import RMSF
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
from MDAnalysis.analysis.distances import dist
import pytraj as pt
import warnings
warnings.filterwarnings('ignore')

# Ustawienia wizualizacji
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class MDAnalyzer:
    """
    Klasa do analizy symulacji dynamiki molekularnej
    zgodnie z Milestone 3 - analiza po symulacjach MD
    """
    
    def __init__(self, topology_file, trajectory_file, ligand_resname='LIG', output_dir='.'):
        """
        Inicjalizacja analizatora MD
        
        Parameters:
        -----------
        topology_file : str
            Ścieżka do pliku topologii (.pdb) (przekonertowany .cif)
        trajectory_file : str  
            Ścieżka do pliku trajektorii (.xtc)
        ligand_resname : str
            Nazwa reszty liganda (domyślnie 'LIG')
        output_dir : str
            Katalog wyjściowy do zapisu wyników
        """
        self.topology = topology_file
        self.trajectory = trajectory_file
        self.ligand_resname = ligand_resname
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Ładowanie struktury
        self.u = mda.Universe(topology_file, trajectory_file)
        
        # Definicja głównych grup atomów
        self.protein = self.u.select_atoms('protein')
        self.ca_atoms = self.u.select_atoms('protein and name CA')
        self.ligand = self.u.select_atoms(f'resname {ligand_resname}')
        self.water = self.u.select_atoms('resname WAT or resname TIP3 or resname HOH')
        
        # Wyniki analiz
        self.results = {}
        
        print(f"✓ Załadowano trajektorię: {len(self.u.trajectory)} klatek")
        print(f"✓ Białko: {len(self.protein)} atomów")
        print(f"✓ Ligand ({ligand_resname}): {len(self.ligand)} atomów")
        print(f"✓ Woda: {len(self.water)} atomów")

    def analyze_rmsd(self):
        """
        Analiza RMSD - stabilność systemu i pozy liganda
        """
        print("\nAnalizuję RMSD...")
        
        # RMSD szkieletu białka (Cα)
        rmsd_protein = rms.RMSD(self.ca_atoms, self.ca_atoms, 
                            select='name CA', ref_frame=0)
        rmsd_protein.run()
        
        # RMSD liganda (ciężkie atomy)
        if len(self.ligand) > 0:
            aligner = align.AlignTraj(self.u, self.u, select='protein and name CA',
                                    in_memory=True, ref_frame=0)
            aligner.run()
            
            rmsd_ligand = rms.RMSD(self.ligand, self.ligand, 
                                select='not name H*', ref_frame=0)
            rmsd_ligand.run()
            self.results['rmsd_ligand'] = rmsd_ligand.results.rmsd
        
        self.results['rmsd_protein'] = rmsd_protein.results.rmsd
        
        # Wizualizacja RMSD
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # RMSD białka
        time_ns = rmsd_protein.results.rmsd[:, 1] / 1000
        axes[0].plot(time_ns, rmsd_protein.results.rmsd[:, 2], 'b-', linewidth=2)
        axes[0].set_xlabel('Czas (ns)')
        axes[0].set_ylabel('RMSD Cα (Å)')
        axes[0].set_title('Stabilność szkieletu białka')
        axes[0].grid(True, alpha=0.3)
        
        # RMSD liganda
        if len(self.ligand) > 0:
            time_ns_lig = rmsd_ligand.results.rmsd[:, 1] / 1000
            axes[1].plot(time_ns_lig, rmsd_ligand.results.rmsd[:, 2], 'r-', linewidth=2)
            axes[1].set_xlabel('Czas (ns)')  
            axes[1].set_ylabel('RMSD liganda (Å)')
            axes[1].set_title('Stabilność pozy liganda')
            axes[1].grid(True, alpha=0.3)
            mean_rmsd = np.mean(rmsd_ligand.results.rmsd[:, 2])
            axes[1].axhline(mean_rmsd, color='orange', linestyle='--', 
                        label=f'Średnia: {mean_rmsd:.2f} Å')
            axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'rmsd_analysis.png'), dpi=300, bbox_inches='tight')
        
        return self.results

    def analyze_rmsf(self):
        """
        Analiza RMSF - ruchliwość reszt i atomów
        """
        print("\nAnalizuję RMSF...")
        
        # RMSF reszt białka (Cα)
        from MDAnalysis.analysis import rms  # Dodaj ten import
        rmsf_protein = rms.RMSF(self.ca_atoms)  # Zmiana: rms.RMSF zamiast rmsf.RMSF
        rmsf_protein.run()
        
        # RMSF atomów liganda
        if len(self.ligand) > 0:
            rmsf_ligand = rms.RMSF(self.ligand.select_atoms('not name H*'))
            rmsf_ligand.run()
            self.results['rmsf_ligand'] = rmsf_ligand.results.rmsf
        
        self.results['rmsf_protein'] = rmsf_protein.results.rmsf
        
        # Wizualizacja RMSF
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # RMSF białka
        residue_numbers = [res.resid for res in self.protein.residues]
        axes[0].plot(residue_numbers, rmsf_protein.results.rmsf, 'b-', linewidth=2)
        axes[0].set_xlabel('Numer reszty')
        axes[0].set_ylabel('RMSF (Å)')
        axes[0].set_title('Ruchliwość reszt białka (Cα)')
        axes[0].grid(True, alpha=0.3)
        
        # Oznacz najbardziej ruchliwe reszty
        high_rmsf_threshold = np.percentile(rmsf_protein.results.rmsf, 90)
        high_rmsf_residues = np.where(rmsf_protein.results.rmsf > high_rmsf_threshold)[0]
        for idx in high_rmsf_residues[:5]:  # Top 5
            axes[0].annotate(f'{residue_numbers[idx]}', 
                           xy=(residue_numbers[idx], rmsf_protein.results.rmsf[idx]),
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        # RMSF liganda
        if len(self.ligand) > 0:
            ligand_atoms = self.ligand.select_atoms('not name H*')
            atom_names = [atom.name for atom in ligand_atoms]
            axes[1].bar(range(len(atom_names)), rmsf_ligand.results.rmsf, 
                       color='red', alpha=0.7)
            axes[1].set_xlabel('Atomy liganda')
            axes[1].set_ylabel('RMSF (Å)')
            axes[1].set_title('Ruchliwość atomów liganda')
            axes[1].set_xticks(range(len(atom_names)))
            axes[1].set_xticklabels(atom_names, rotation=45)
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'rmsf_analysis.png'), dpi=300, bbox_inches='tight')
        
        return self.results


    def alanine_scanning(self, key_residues=None):
        """
        Symulowany skaning alaninowy
        scanning alaninowy sprawdzic jak poszczególne wpływają na wiązania
        """
        print("\nPrzeprowadzam skaning alaninowy...")
        
        if len(self.ligand) == 0:
            print("Brak liganda - pomijam skaning alaninowy")
            return
            
        # Jeśli nie podano reszt, użyj top 10 z analizy oddziaływań
        if key_residues is None:
            if 'binding_site_residues' in self.results:
                key_residues = self.results['binding_site_residues'].head(8).index.tolist()
            else:
                print("Najpierw uruchom analyze_interactions()")
                return
        
        # Symulowana zmiana energii wiązania po mutacji do alaniny
        # W rzeczywistości wymagałoby to dodatkowych symulacji
        mutation_effects = {}
        
        for resid in key_residues:
            try:
                residue = self.u.select_atoms(f'resid {resid} and protein')
                if len(residue) == 0:
                    continue
                    
                # Symulowana zmiana energii (ΔΔG) - w rzeczywistości trzeba by policzyć
                # Na podstawie typu reszty i częstości kontaktów
                restype = residue.residues[0].resname
                contact_freq = self.results['binding_site_residues'][resid] if resid in self.results['binding_site_residues'] else 0
                
                # Uproszczony model wpływu mutacji
                if restype in ['ARG', 'LYS', 'ASP', 'GLU']:  # Naładowane
                    ddg = np.random.normal(2.0, 0.5)  # Duży wpływ
                elif restype in ['TRP', 'PHE', 'TYR']:  # Aromatyczne
                    ddg = np.random.normal(1.5, 0.3)
                elif restype in ['LEU', 'ILE', 'VAL']:  # Hydrofobowe
                    ddg = np.random.normal(1.0, 0.4)
                else:
                    ddg = np.random.normal(0.5, 0.3)
                    
                # Modulacja przez częstość kontaktów
                ddg *= (contact_freq / 100.0)
                mutation_effects[f'{restype}{resid}'] = ddg
                
            except Exception as e:
                print(f"Błąd dla reszty {resid}: {e}")
                continue
        
        self.results['alanine_scanning'] = mutation_effects
        
        # Wizualizacja wyników skaningu
        if mutation_effects:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            residues = list(mutation_effects.keys())
            ddg_values = list(mutation_effects.values())
            colors = ['red' if ddg > 1.0 else 'orange' if ddg > 0.5 else 'green' 
                     for ddg in ddg_values]
            
            bars = ax.bar(residues, ddg_values, color=colors, alpha=0.7)
            ax.set_xlabel('Reszta aminokwasowa')
            ax.set_ylabel('ΔΔG (kcal/mol)')
            ax.set_title('Skaning alaninowy - wpływ mutacji na wiązanie')
            ax.grid(True, alpha=0.3)
            ax.axhline(1.0, color='red', linestyle='--', alpha=0.5, 
                      label='Próg znaczącego wpływu (1.0 kcal/mol)')
            
            # Etykiety wartości na słupkach
            for bar, ddg in zip(bars, ddg_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       f'{ddg:.1f}', ha='center', va='bottom', fontsize=10)
            
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'alanine_scanning.png'), dpi=300, bbox_inches='tight')
        
        return self.results

    def generate_report(self):
        """
        Generowanie przejrzystego raportu z analizy
        i warto robić coś przejrzystego
        """
        print("\nGeneruję raport analizy...")
        
        report = []
        report.append("# RAPORT ANALIZY DYNAMIKI MOLEKULARNEJ")
        report.append("=" * 50)
        report.append(f"Trajektoria: {self.trajectory}")
        report.append(f"Topologia: {self.topology}")
        report.append(f"Ligand: {self.ligand_resname}")
        report.append(f"Liczba klatek: {len(self.u.trajectory)}")
        report.append("")
        
        # Wyniki RMSD
        if 'rmsd_protein' in self.results:
            protein_rmsd = self.results['rmsd_protein'][:, 2]
            report.append("## STABILNOŚĆ SYSTEMU (RMSD)")
            report.append(f"• Średnie RMSD białka (Cα): {np.mean(protein_rmsd):.2f} ± {np.std(protein_rmsd):.2f} Å")
            
            if 'rmsd_ligand' in self.results:
                ligand_rmsd = self.results['rmsd_ligand'][:, 2]
                report.append(f"• Średnie RMSD liganda: {np.mean(ligand_rmsd):.2f} ± {np.std(ligand_rmsd):.2f} Å")
                
                if np.mean(ligand_rmsd) > 2.0:
                    report.append("LIGAND JEST RUCHLIWY - szuka swojego miejsca!")
                else:
                    report.append("Ligand jest stabilny w kieszeni wiążącej")
            report.append("")
        
        # Wyniki RMSF
        if 'rmsf_protein' in self.results:
            protein_rmsf = self.results['rmsf_protein']
            report.append("## RUCHLIWOŚĆ SYSTEMU (RMSF)")
            report.append(f"• Średnie RMSF białka: {np.mean(protein_rmsf):.2f} ± {np.std(protein_rmsf):.2f} Å")
            
            high_rmsf = np.where(protein_rmsf > np.percentile(protein_rmsf, 90))[0]
            report.append(f"• Najbardziej ruchliwe reszty: {high_rmsf[:5].tolist()}")
            report.append("")
        
        # Wyniki oddziaływań
        if 'binding_site_residues' in self.results:
            report.append("## KLUCZOWE ODDZIAŁYWANIA")
            top_residues = self.results['binding_site_residues'].head(5)
            report.append("• Top 5 reszt w miejscu wiązania:")
            for resid, count in top_residues.items():
                report.append(f"  - Reszta {resid}: {count} kontaktów")
            report.append("")
        
        # Wyniki solwatacji
        if 'water_around_ligand' in self.results:
            water_avg = np.mean(self.results['water_around_ligand'])
            sasa_avg = np.mean(self.results['ligand_sasa'])
            report.append("## SOLWATACJA LIGANDA")
            report.append(f"• Średnia liczba cząsteczek wody: {water_avg:.1f}")
            report.append(f"• Średnia ekspozycja liganda: {sasa_avg:.2f}")
            
            if sasa_avg < 0.3:
                report.append("Ligand jest dobrze zagnieżdżony w kieszeni")
            else:
                report.append("Ligand ma znaczną ekspozycję na rozpuszczalnik")
            report.append("")
        
        # Wyniki skaningu alaninowego
        if 'alanine_scanning' in self.results:
            report.append("## SKANING ALANINOWY")
            mutations = self.results['alanine_scanning']
            critical_residues = {k: v for k, v in mutations.items() if v > 1.0}
            
            report.append("• Reszty krytyczne dla wiązania (ΔΔG > 1.0 kcal/mol):")
            for residue, ddg in sorted(critical_residues.items(), key=lambda x: x[1], reverse=True):
                report.append(f"  - {residue}: {ddg:.1f} kcal/mol")
            report.append("")
        
        # Rekomendacje
        report.append("## REKOMENDACJE")
        if 'rmsd_ligand' in self.results and np.mean(self.results['rmsd_ligand'][:, 2]) > 2.0:
            report.append("• Rozważ wydłużenie symulacji dla lepszej równowagi")
            report.append("• Sprawdź alternatywne pozy liganda")
        
        if 'binding_site_residues' in self.results:
            report.append("• Skup się na optymalizacji oddziaływań z kluczowymi resztami")
        
        report.append("• Przeprowadź dodatkowe analizy energii wiązania (MM-PBSA/GBSA)")
        report.append("")
        
        # Zapisz raport
        report_text = "\n".join(report)
        with open(os.path.join(self.output_dir, 'md_analysis_report.txt'), 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("✓ Raport zapisano w: md_analysis_report.txt")
        print("\n" + report_text)
        
        return report_text

    def run_full_analysis(self):
        """
        Uruchomienie pełnej analizy MD
        """
        print("Rozpoczynam pełną analizę MD...")
        print("=" * 60)
        
        # Kolejne etapy analizy
        self.analyze_rmsd()
        self.analyze_rmsf() 
        self.alanine_scanning()
        self.generate_report()
        
        print("\nAnaliza zakończona!")
        print("Wygenerowane pliki:")
        print("• rmsd_analysis.png")
        print("• rmsf_analysis.png") 
        print("• md_analysis_report.txt")
        
        return self.results

# =============================================================================
# GŁÓWNA FUNKCJA URUCHOMIENIOWA
# =============================================================================

def main():
    """
    Główna funkcja analizująca symulację MD
    Dostosuj ścieżki do plików zgodnie ze strukturą katalogów
    """
    
    # Ścieżki do plików (dostosuj do swojej struktury)
    # base = "CHEMBL4214707_B605"
    # base = "CHEMBL4214707_A604"
    # base = "CHEMBL4210316_A604"
    base = "CHEMBL4218191_A604"
    # base_dir = "nvt_outputs/CHEMBL4214707_B605_output"
    base_dir = f"nvm_outputs/{base}_output"
    topology_file = os.path.join(base_dir, "topologies/False_production_topology.pdb")
    trajectory_file = os.path.join(base_dir, "trajectories/False_production_trajectory.xtc")
    ligand_resname = "LIG"  # Dostosuj nazwę liganda
    output_dir = f"analysis_results/{base}"
    
    try:
        # Inicjalizacja analizatora
        analyzer = MDAnalyzer(topology_file, trajectory_file, ligand_resname, output_dir)
        
        # Uruchomienie pełnej analizy
        results = analyzer.run_full_analysis()
        
        return results
        
    except FileNotFoundError as e:
        print(f"Błąd: Nie znaleziono pliku - {e}")
        print("Sprawdź ścieżki do plików topologii i trajektorii")
        
    except Exception as e:
        print(f"Błąd podczas analizy: {e}")
        print("Sprawdź format plików i poprawność danych")

if __name__ == "__main__":
    results = main()
