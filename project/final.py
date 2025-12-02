#===============================================================================
# PROGRAM : final.py -> Final Submission
# PURPOSE : Develop a webapp that explores datasets and showcases understanding
#           of material from class. I have decided to explore nuclear physics
#           datasets.
# DATE CREATED  : 30.09.25
# LAST MODIFIED : 15.10.25
#
# AUTHOR*     : Joshua Belieu | Fletch
#                *Portions of this code were written or augmented with an LLM.
#                 Sections are marked adherent to AIPolicy.
# EMAIL       : belieujo@msu.edu
# DEPARTMENT  : Physics & Astronomy
# INSTITUTION : Michigan State University | Facility for Rare Isotope Beams
#
# EXECUTE PROGRAM : streamlit run final.py
#===============================================================================
#                                BEGIN PROGRAM                                 
#-------------------------------------------------------------------------------

#*******************************************************************************
#                               IMPORTED LIBRARIES
#*******************************************************************************

import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
import plotly.express as px
import os
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import PCA

#*******************************************************************************
#                            LOAD AND FUTZ THE DATA
#******************************************************************************* 

# pathing error for app deployment. ChatGPT 5.0 (14.10.25) reccomended this fix.
# Get the directory where this file (app.py) lives
APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Build full path to data file
ame_path = os.path.join(APP_DIR, "ame2020.txt")
rc_path = os.path.join(APP_DIR, "charge_radius.csv")
b2_path = os.path.join(APP_DIR, "nndc_nudat_data_export.csv")

def load_in_ame2020 ( filename = ame_path ) :

    # Define the column widths based on the Fortran format
    colspecs = [
        (0, 1),      # cc (1 char)
        (1, 4),      # NZ (i3)
        (4, 9),      # N (i5)
        (9, 14),     # Z (i5)
        (14, 19),    # A (i5)
        (19, 20),    # space
        (20, 23),    # EL (a3)
        (23, 27),    # O (a4)
        (27, 28),    # space
        (28, 42),    # mass excess (f14.6)
        (42, 54),    # mass uncertainty (f12.6)
        (54, 67),    # binding energy (f13.5)
        (67, 68),    # space
        (68, 78),    # B (f10.5)
        (78, 79),    # space
        (79, 81),    # beta (a2)
        (81, 94),    # beta uncertainty (f13.5)
        (94, 105),   # atomic mass (f11.5)
        (105, 106),  # space
        (106, 109),  # integer (i3)
        (109, 110),  # space
        (110, 123),  # f13.6
        (123, 135),  # f12.6
    ]

    # Column names (adapt to your preference)
    colnames = [
        "cc", "1N-Z", "n", "z", "a", "space1", "EL", "O", "space2",
        "mass_excess", "mass_unc", "binding_energy", "space3",
        "bind_ener_unc", "space4", "beta_mode", "beta_ener","beta_unc" ,"space5", "space6",
        "space7", "atomic_mass", "atom_mass_unc"
    ]

    # Read the fixed-width file
    df_ame = pd.read_fwf(filename, colspecs=colspecs, names=colnames, skiprows=36)

    # Optionally, drop the space columns
    df_ame = df_ame.drop(columns=[c for c in colnames if "space" in c])
    df_ame = df_ame.drop(columns='cc')

    # Convert numeric columns to floats
    numeric_cols = ["1N-Z", "n", "z", "a", "mass_excess", "mass_unc", "binding_energy",
                    "bind_ener_unc","beta_ener" ,"beta_unc", "atomic_mass", "atom_mass_unc"]
    for col in numeric_cols:
        df_ame[col] = pd.to_numeric(df_ame[col], errors="coerce")

    return df_ame

df_ame = load_in_ame2020()
df_rc = pd.read_csv(rc_path)
df_b2 = pd.read_csv (b2_path)
df_merge_temp = pd.merge(df_ame , df_rc , on = ["z", "n"] , how = "outer" )
df_merge = pd.merge(df_merge_temp , df_b2 , on = ["z", "n"] , how = "outer" )

# ChatGPT wrote this handler for when pandas notices repeat columns in a merger.
# (10.10.2025)
# If both dataframes had an 'a' column, pandas created 'a_x' and 'a_y'
if 'a_x' in df_merge.columns and 'a_y' in df_merge.columns:
    # Prefer 'a_x' if it exists, otherwise use 'a_y'
    df_merge['a'] = df_merge['a_x'].combine_first(df_merge['a_y'])
    df_merge = df_merge.drop(columns=['a_x', 'a_y'])

#*******************************************************************************
#                             STREAMLIT PAGES SETUP
#*******************************************************************************
 
st.sidebar.title('Navigation')

page_options = ["Home Page","Introduction to Nuclear Physics",
                "Datasets","Exploring Correlations","Interactive Plot","Machine Learning Applications"]

page = st.sidebar.radio ( "Go to" , page_options )

if page == page_options [ 0 ] : # home page

    st.image('https://educationusa.state.gov/sites/default/files/field_hei_logo/msu_logo.png',width=1e3)

    st.title ( 'Home Page' )
    st.header ( 'CMSE 830 : Foundations of Data Science Semester Project' )
    st.markdown ( '''Hello! Welcome to my project that approaches and investigates
        three nuclear physics datasets from a data science perspective. Hopefully,
        I can employ data science techniques to 'discover' common nuclear 
        physics notions or maybe even find something new!  
        Nuclear physics is cool but complex and difficult to follow at times.
        Please see my 'introduction to nuclear physics' tab (found on  the left
        side of your screen) for a light explanation of topics you will find on
        this app.''' )

if page == page_options [ 1 ] : # intro to nucl. phys.

    st.image ( 'https://physicsde.video.blog/wp-content/uploads/2018/10/dd0cafce-a2b8-4648-84bd5c8d43e8e24a.jpg?w=426&h=220' )

    st.title("A *Brief* Introduction to Nuclear Physics")

    st.markdown("""
    Nuclear physics is notoriously difficult to understand and study. While we 
    cannot explore the subject deeply here, it will behoove us to gain a working
    understanding for what is to come.  

    Matter is made up of elements composed of atoms. Atoms are generally thought
    of as electrons orbiting a central core of particles, the "nucleus". The 
    nucleus is made up of protons and neutrons. As nuclear physicists, we 
    typically only care about the nuclei, for which we use the following 
    symbols:
    """)

    st.markdown(r"""
    $z$ — "Proton number" : The number of protons a nucleus has.  

    $n$ — "Neutron number" : The number of neutrons a nucleus has.  

    $a \equiv n + z$ — "Nucleon number" : Total number of protons and neutrons.
    """)

    st.markdown(r"""
    One of the first questions we can ask about a nucleus is *"How heavy is it?"*

    The most intuitive answer might be to claim that the mass of a nucleus is 
    simply the number of protons multiplied by the mass of one proton, plus the
    number of neutrons multiplied by the mass of one neutron:

    $$
    m_N = z\times m_p + n\times m_n
    $$

    However, experiments show a surprising result: the measured mass of a 
    nucleus is actually *less* than this simple calculation predicts. This 
    difference is known as the **mass defect**, $\Delta m$.
                
    To account for this mass defect, physcists went back to drawing board and 
    made an interesting claim: *The nucleus is a **self bound** system*. That 
    is, there is something that is trying to keep the protons and neutrons close
    together. This is known commonly as the nuclear force. 
                
    After some thought physicists at the time decided to change their nuclear 
    mass equation to account for this mass defect and defined the *nuclear 
    binding energy*, $BE$ which they guessed was a function of the number of 
    protons and neutrons a nucleus would have:

    $$
    m_N = z\times m_p + n\times m_n - BE(n,z)
    $$
                
    The next question one might ask is *"How big is a nucleus?"*

    **This is where it gets complicated.** Experimentally, physicists measure 
    the size of a nucleus by measuring energy levels of light nuclei and how 
    electrons scatter off nuclons, this typically referred to as the *charge 
    radius*, $r_c$. We could discuss the intricacies of how complex and nuanced
    this answer could be from a theoretical perspective but instead lets leave 
    it as an exercise for the user to answer the question, *"What determines how big a nucleus
    is?"* See the 'interactive plot' tab to see more!

    In nuclear physics, we also like to talk about how a nucleus might change its composition. 
    This is commonly referred to as $\beta$-decay. There are a few ways in which this might 
    occur but for the purposes of this project we will only talk about the process known as
    $\beta^-$ decay. The process is seen as
    
    $$
    (n,z) \rightarrow (n-1,z+1) + e^- + \bar{\nu}_e
    $$
                
    That is, a nucleus with $(n,z)$ particles changes into a nucleus with $(n-1,z+1)$ where an electron and 
    an anti-electron neutrino are emitted in the process (don't worry about these, but they're cool!).

    The $\beta$-decay energy, $Q$, represents the amount of energy released in this transformation.  
    It is calculated from the atomic masses of the parent and daughter nuclei as:

    $$
    Q_{-} = [M(n,z) - M(n-1,z+1)]c^2
    $$
                
    One final question we might ask about nuclei is **What shape are they?** this is yet another complicated
    question (I'm noticing a trend forming here) so we will leave it here to just say that physicists discuss
    the shape of nuclei through the quadrupole deformation parameter, $$\beta_2$$. This parameter has a relatively
    simple interpretation:
                
    $$
    \beta_2\rightarrow\begin{cases} >0 & \text{The nucleus is taller than it is wide, prolate.} \\
    =0 & \text{The nucleus is spherical.} \\
    <0 & \text{The nucleus is wider than it is tall, oblate.}\end{cases}          
    $$
                
    This is shown here nicely:""")
                
    st.image ( "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSkwWeCSosvA7wO6djTM549_szrXOhm_UQzcQ&s" )

    st.markdown(r"""

    Physicists, like data scientists, often look for convenient ways to package
    information so that it is convenient to analyze. Here we will look at two
    such ways. First we ask, *"Is there anything special about a nucleus that 
    has an odd or even number of nucleons?"* To answer answer this we want to 
    consider the **parity, $\Pi$** of a nucleus. Parity is classified via:
                

    $$
    \Pi = 
    \begin{cases} 
    \text{Even-Even}, & \text{if $n$ and $z$ are both even} \\
    \text{Even-Odd / Odd-Even}, & \text{if one of $n$ or $z$ is even, the other odd} \\
    \text{Odd-Odd}, & \text{if both $n$ and $z$ are odd}
    \end{cases}
    $$
                
    The second manner in which we mark data is an answer to the question, *"How
    many more neutrons are there than protons?"* This is given a fancy name
    known as **Isospin Asymmetry**,I.
                
    $$
    I=\frac{n-z}{n+z}=\frac{n\pm z-z}{a}=\frac{n+z-2z}{a}=\frac{a-2z}{a}
    =1-2\frac{z}{a}\equiv 1-2y_p
    $$
                
    Where this last expression introduces the 'proton fraction', $y_p$.
    """)

if page == page_options [ 2 ] : # datasets

    st.title ( 'Datasets' )

    st.header( 'Lets take a look at our datasets!' )

    dataset_options = [ 'Binding Energy' ,
                        'Charge Radii' ,
                        'Quadrupole Deformation' ,
                        'Merged ( Binding Energy + Charge Radii + Quadrupole Deformations)' ]

    dataset_choice = st.radio ( 'Select a dataset to view : ' ,
                                dataset_options ,
                                horizontal = True )
    
    if dataset_choice == dataset_options [ 0 ] : # Binding Energy
        df_ds = df_ame.copy()
        default_option = [ 'mass_excess' ,
                    'binding_energy' ,
                    'beta_ener' ,
                    'atomic_mass' ]
    if dataset_choice == dataset_options [ 1 ] : # Charge Radii
        df_ds = df_rc.copy()
        default_option = [ 'radius_val' ]
    if dataset_choice == dataset_options [ 2 ] : # quad def
        df_ds = df_b2.copy()
        default_option = [ 'radius_val' ]
    if dataset_choice == dataset_options [ 3 ] : # Merged
        df_ds = df_merge.copy()
        default_option = [ 'mass_excess' ,
                      'binding_energy' ,
                      'beta_ener' ,
                      'atomic_mass' ,
                      'radius_val' ]

    scaling_options = [ 'None' , 'Standard' , 'Min/Max']
    scaling_string = 'Would you like to apply a Scaler to the data?'
    scaling_choice = st.radio ( scaling_string ,
                               scaling_options ,
                               horizontal = True )

    # I wrote some code to apply a scaler to the data and recombine it all for 
    # plotting and such. After, I asked chatGPT 5.0 if there was way to do it 
    # better and it generated this. (08.10.25)

    # Apply scaling if selected
    if scaling_choice != 'None':
        numeric_cols = df_ds.select_dtypes(include=["number"]).copy()

        cols_to_scale = st.multiselect(
            "Select columns to apply scaling to:",
            options=numeric_cols.columns.tolist(),
            default=default_option
        )

        if scaling_choice == 'Standard' and cols_to_scale:
            scaler = StandardScaler()
            numeric_cols[cols_to_scale] = scaler.fit_transform(
                numeric_cols[cols_to_scale])
        elif scaling_choice == 'Min/Max' and cols_to_scale:
            scaler = MinMaxScaler()
            numeric_cols[cols_to_scale] = scaler.fit_transform(
                numeric_cols[cols_to_scale])

        non_numeric_cols = df_ds.select_dtypes(exclude=["number"]).copy()
        df_ds = pd.concat([non_numeric_cols, numeric_cols], axis=1)

    else:
        df_scaled = df_ds.copy()

    impute_options = [ 'No' , 'Drop NaNs' , 'Impute Mean' ,
                       'Impute Median' , 'Impute Mode' ]
    impute_string = 'The dataset might be dirty, would you like to do' \
                    ' anyhting about that?'
    impute_choice = st.radio ( impute_string , 
                               impute_options , 
                               horizontal = True )
    
    if impute_choice == impute_options [ 0 ] : # No
        st.write ( 'Brave choice! be aware that things may break!' )
    elif impute_choice == impute_options [ 1 ] : # drop nans
        df_ds = df_ds.dropna()
    elif impute_choice == impute_options [ 2 ] : # impute mean
        df_ds = df_ds.fillna ( df_ds.mean(numeric_only=True) )
    elif impute_choice == impute_options [ 3 ] : # impute median
        df_ds = df_ds.fillna ( df_ds.median(numeric_only=True) )
    elif impute_choice == impute_options [ 4 ] : # impute mode
        df_ds = df_ds.fillna ( df_ds.mode(numeric_only=True).iloc[0] )

    description_options = [ 'Data Information' , 'Summary Statisitics' , 
                           'Missingness Visualzation' ]
    description_string = f'What about the {dataset_choice} dataset would you like to see?'

    st.markdown( description_string )
    
    data_info_toggle = st.toggle ( description_options [ 0 ] )
    data_stats_toggle = st.toggle ( description_options [ 1 ] )
    data_miss_toggle = st.toggle ( description_options [ 2 ] )
    data_pca_toggle = st.toggle ( 'PCA' )

    if data_info_toggle == True :
        st.write("**Shape:**", df_ds.shape)
        st.write("**Columns:**", df_ds.columns)
    if data_stats_toggle == True :
        st.dataframe(df_ds.describe())
    if data_miss_toggle == True : # ChatGPT 5.0 helped me with this bit. 
                                  # specifically, the table. (08.10.2025)

        missing_counts = df_ds.isnull().sum()
        missing_pct = (missing_counts / len(df_ds)) * 100
        missing_df = pd.DataFrame({
            "Missing Values Count": missing_counts,
            "Missing Values Percentage": missing_pct
        }).sort_values("Missing Values Percentage", ascending=False)

        st.markdown("### Missing Data Table")

        st.dataframe(missing_df.style.background_gradient(cmap='viridis'))

        st.markdown("### Missing Data Heatmap")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.set(font_scale=1.4)
        sns.heatmap(df_ds.isnull().transpose(), xticklabels=False,cmap='viridis')
        st.pyplot(fig)

    # st.download_button(f"Download {dataset_choice}? Click me!",
    #                    df_ds.to_csv(index=False), 
    #                    file_name=f"{dataset_choice}.csv")

    if data_pca_toggle == True : # pca

        st.header("PCA Visualization")

        # Copy the dataset for PCA
        df_pca = df_ds.copy()
        df_pca['isospin_asymmetry'] = ( df_pca['n'] - df_pca['z'] ) / ( df_pca['n'] + df_pca['z'] )
        parity_conditions = [
        (df_pca['n'] % 2 == 0) & (df_pca['z'] % 2 == 0), # is n&z even?
        ((df_pca['n'] % 2 != 0) & (df_pca['z'] % 2 == 0)) | # is n&z even&odd or...  
        ((df_pca['n'] % 2 == 0) & (df_pca['z'] % 2 != 0)), # vice versa?
        (df_pca['n'] % 2 != 0) & (df_pca['z'] % 2 != 0) # is n&z odd?
        ]

        parity_labels = ['Even-Even', 'Even-Odd', 'Odd-Odd']
        df_pca['parity'] = np.select(parity_conditions, parity_labels,default=str)


        # Let user select features for PCA
        numeric_cols = df_pca.select_dtypes(include=np.number).columns.tolist()
        selected_features = st.multiselect(
            "Select numeric features to include in PCA:",
            options=numeric_cols,
            default=numeric_cols
        )

        if len(selected_features) < 2:
            st.warning("Select at least 2 features for PCA.")
        else:
            df_pca_selected = df_pca[selected_features].copy()

            if scaling_choice != 'None':
                scaler = StandardScaler() if scaling_choice == 'Standard' else MinMaxScaler()
                df_pca_selected = pd.DataFrame(
                    scaler.fit_transform(df_pca_selected),
                    columns=df_pca_selected.columns,
                    index=df_pca_selected.index
                )

            # PCA computation
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(df_pca_selected)
            df_plot = pd.DataFrame(pca_result, columns=['PC1', 'PC2'], index=df_pca_selected.index)

            # Color selection
            color_options = ['None'] + df_pca.select_dtypes(include=['number', 'object']).columns.tolist()
            color_choice = st.selectbox("Color points by:", color_options)

            if color_choice != 'None':
                df_plot[color_choice] = df_pca.loc[df_pca_selected.index, color_choice].values

            # Plot PCA
            fig_pca = px.scatter(
                df_plot,
                x='PC1',
                y='PC2',
                color=color_choice if color_choice != 'None' else None,
                title='PCA Scatter Plot',
                labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'}
            )
            st.plotly_chart(fig_pca, use_container_width=True)

            st.header("PCA Biplot")

            df_biplot = df_ds.copy()
            df_biplot['isospin_asymmetry'] = (df_biplot['n'] - df_biplot['z']) / (df_biplot['n'] + df_biplot['z'])

            numeric_cols = df_biplot.select_dtypes(include=np.number).columns.tolist()
            selected_features = st.multiselect(
                "Select numeric features for PCA (Biplot):",
                options=numeric_cols,
                default=numeric_cols
            )

            if len(selected_features) < 2:
                st.warning("Select at least 2 features for PCA.")
            else:
                df_biplot_selected = df_biplot[selected_features].dropna()

                # Standardize
                scaler = StandardScaler()
                df_scaled = pd.DataFrame(scaler.fit_transform(df_biplot_selected),
                                        columns=df_biplot_selected.columns,
                                        index=df_biplot_selected.index)

                # PCA
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(df_scaled)
                df_plot = pd.DataFrame(pca_result, columns=['PC1', 'PC2'], index=df_scaled.index)

                # Scatter plot
                fig_biplot = px.scatter(
                    df_plot,
                    x='PC1',
                    y='PC2',
                    opacity=0.7,
                    title='PCA Biplot',
                    labels={'PC1': 'PC1', 'PC2': 'PC2'}
                )

                # Loadings
                loadings = pca.components_.T
                # Scale arrows to fit scatter plot nicely
                arrow_scale = 3  # tweak this factor
                loadings_scaled = loadings * arrow_scale

                for i, feature in enumerate(df_scaled.columns):
                    fig_biplot.add_annotation(
                        x=loadings_scaled[i, 0],
                        y=loadings_scaled[i, 1],
                        ax=0,
                        ay=0,
                        xref="x",
                        yref="y",
                        axref="x",
                        ayref="y",
                        showarrow=True,
                        arrowhead=3,
                        arrowsize=1.5,
                        arrowwidth=2,
                        arrowcolor="red",
                        text="",  # hide text on the arrow itself
                    )
                    # Add label slightly offset from arrow tip
                    fig_biplot.add_annotation(
                        x=loadings_scaled[i, 0] * 1.1,  # offset 10%
                        y=loadings_scaled[i, 1] * 1.1,
                        text=feature,
                        showarrow=False,
                        font=dict(color="red", size=12)
                    )

                st.plotly_chart(fig_biplot, use_container_width=True)


if page == page_options [ 3 ] : # exploring correlations

    st.title ( 'Exploring Correlations' )
    st.markdown ( 'Lets explore correlations in our datasets!' )

    ec_options = [ 'Correlation Heatmap' , 'Pairplot' ,  ]
    ec_choice = st.radio ( 'How would you like to view your correlations?' , ec_options )

    dataset_options = [ 'Binding Energy' ,
                        'Charge Radii' ,
                        'Quadrupole Deformations' ,
                        'Merged ( Binding Energy + Charge Radii )' ]
    dataset_choice = st.radio ( 'Select a dataset to explore : ' ,
                               dataset_options ,
                               horizontal = True ) 

    if dataset_choice == dataset_options [ 0 ] :
        df_ec = df_ame.copy()
        default_option = [ 'mass_excess' ,
                    'binding_energy' ,
                    'beta_ener' ,
                    'atomic_mass' ]
    if dataset_choice == dataset_options [ 1 ] :
        df_ec = df_rc.copy()
        default_option = [ 'radius_val' ]
    if dataset_choice == dataset_options [ 2 ] :
        df_ec = df_b2.copy()
        df_ec['a']=df_ec['n']+df_ec['z']
        default_option = [ 'quadrupoleDeformation' ]
    if dataset_choice == dataset_options [ 3 ] :
        df_ec = df_merge.copy()
        default_option = [ 'mass_excess' ,
                      'binding_energy' ,
                      'beta_ener' ,
                      'atomic_mass' ,
                      'radius_val' ]

    df_ec['isospin_asymmetry'] = ( df_ec['n'] - df_ec['z'] ) / ( df_ec['n'] + df_ec['z'] )

    
    # I wrote an encoding routine that accomplishes the below but I used a for 
    # loop which I recall being told is slow. So I dropped it into ChatGPT 5.0
    # and asked 'can this be done better?' and it output this robust bool
    # method. (08.10.2025)
    
    parity_conditions = [
    (df_ec['n'] % 2 == 0) & (df_ec['z'] % 2 == 0), # is n&z even?
    ((df_ec['n'] % 2 != 0) & (df_ec['z'] % 2 == 0)) | # is n&z even&odd or...  
    ((df_ec['n'] % 2 == 0) & (df_ec['z'] % 2 != 0)), # vice versa?
    (df_ec['n'] % 2 != 0) & (df_ec['z'] % 2 != 0) # is n&z odd?
    ]

    parity_labels = ['Even-Even', 'Even-Odd', 'Odd-Odd']
    df_ec['parity'] = np.select(parity_conditions, parity_labels,default=str)

    impute_options = [ 'No' , 'Drop NaNs' , 'Impute Mean' ,
                       'Impute Median' , 'Impute Mode' ]
    impute_string = 'The dataset might be dirty, would you like to do' \
                    ' anyhting about that?'
    impute_choice = st.radio ( impute_string , 
                               impute_options , 
                               horizontal = True )
    
    if impute_choice == impute_options [ 0 ] : # No
        st.write ( 'Brave choice! be aware that things may break!' )
    elif impute_choice == impute_options [ 1 ] : # drop nans
        df_ec = df_ec.dropna()
    elif impute_choice == impute_options [ 2 ] : # impute mean
        df_ec = df_ec.fillna ( df_ec.mean(numeric_only=True) )
    elif impute_choice == impute_options [ 3 ] : # impute median
        df_ec = df_ec.fillna ( df_ec.median(numeric_only=True) )
    elif impute_choice == impute_options [ 4 ] : # impute mode
        df_ec = df_ec.fillna ( df_ec.mode(numeric_only=True).iloc[0] )

    scaling_options = [ 'None' , 'Standard' , 'Min/Max']
    scaling_string = 'Would you like to apply a Scaler to the data?'
    scaling_choice = st.radio ( scaling_string ,
                               scaling_options ,
                               horizontal = True )

    # I wrote some code to apply a scaler to the data and recombine it all for 
    # plotting and such. After, I asked chatGPT 5.0 if there was way to do it 
    # better and it generated this. (08.10.25)

    # Apply scaling if selected
    if scaling_choice != 'None':
        numeric_cols = df_ec.select_dtypes(include=["number"]).copy()

        cols_to_scale = st.multiselect(
            "Select columns to apply scaling to:",
            options=numeric_cols.columns.tolist(),
            default=default_option
        )

        if scaling_choice == 'Standard' and cols_to_scale:
            scaler = StandardScaler()
            numeric_cols[cols_to_scale] = scaler.fit_transform(
                numeric_cols[cols_to_scale])
        elif scaling_choice == 'Min/Max' and cols_to_scale:
            scaler = MinMaxScaler()
            numeric_cols[cols_to_scale] = scaler.fit_transform(
                numeric_cols[cols_to_scale])

        non_numeric_cols = df_ec.select_dtypes(exclude=["number"]).copy()
        df_ec = pd.concat([non_numeric_cols, numeric_cols], axis=1)

    else:
        df_ec = df_ec.copy()

    # Ensure critical columns exist, i asked ChatGPT what was going on when
    # it the code didnt recognize parity. it output this (10.10.2025)
    for col in ['isospin_asymmetry', 'parity','a']:
        if col not in df_ec.columns:
            df_ec[col] = df_ec[col]

    # Confirm column list for plotting
    available_cols = [c for c in default_option + ['isospin_asymmetry' , 'parity','a'] if c in df_ec.columns]

    # st.write("**Columns available for pairplot:**", available_cols)

    if ec_choice == ec_options [ 0 ] :

        # these commented out lines are the first iteration of the 
        # visualization. after making it, I dropped it into chatGPT and asked
        # if it can be done with plotly. that code is now implemented (15.10.25)
        # sns.reset_defaults()
        numeric_cols = df_ec[available_cols].select_dtypes(include=np.number)
        # heatmap = sns.heatmap(numeric_cols.corr(), annot=True, cmap='viridis')
        # st.pyplot(heatmap.figure)
        # Compute correlation matrix
        corr_matrix = numeric_cols.corr()

        # Create Plotly heatmap
        fig = px.imshow(
            corr_matrix,
            text_auto=True,       # show correlation values on cells
            aspect="auto",        # keep cells square-ish
            color_continuous_scale='Viridis',
            labels=dict(x="Features", y="Features", color="Correlation")
        )

        # Show in Streamlit
        st.plotly_chart(fig, use_container_width=True)

    if ec_choice == ec_options [ 1 ] :

        kind_options = ["Scatter",
                "Regression",
                "Kernel Density Estimator",
                "Histogram"]
        kind_string = 'What KIND of pairplot would you like?'
        kind_choice = st.radio( kind_string , kind_options , horizontal = True )

        if kind_choice == kind_options [ 0 ] :
            kind = 'scatter'
        if kind_choice == kind_options [ 1 ] :
            kind = 'reg'
        if kind_choice == kind_options [ 2 ] :
            kind = 'kde'
        if kind_choice == kind_options [ 3 ] :
            kind = 'hist'

        encoder_options = [ 'None' , 'Isospin Asymmetry' , 'Parity' ]
        encoder_string = 'Select an encoder to color the pairplot by.'
        encoder_choice = st.radio ( encoder_string ,
                                encoder_options ,
                                horizontal = True )

        if encoder_choice == encoder_options [ 0 ] :
            hue = None
        if encoder_choice == encoder_options [ 1 ] :
            hue = 'isospin_asymmetry'
        if encoder_choice == encoder_options [ 2 ] :
            hue = 'parity'

        if st.button("Render Pairplot"):
            with st.spinner("Generating pairplot..."):
                sns.set_context("talk", font_scale=1.6)
                fig = sns.pairplot(
                    df_ec[available_cols],
                    hue=hue,
                    kind=kind,
                    diag_kind='auto',
                    height=4,
                )
                st.pyplot(fig)

if page == page_options [ 4 ] : # interactive plot

    st.title("Interactive Plot")
    st.markdown(r'''
        After looking at the Exploring Correlations tab I saw an interesting 
        correlation between  $r_c$ and $a$! Lets play around and see if we can
        fine tune it.''')
    


    df_ip = df_merge.copy()

    impute_options = [ 'No' , 'Drop NaNs' , 'Impute Mean' ,
                       'Impute Median' , 'Impute Mode' ]
    impute_string = 'The dataset might be dirty, would you like to do' \
                    ' anyhting about that?'
    impute_choice = st.radio ( impute_string , 
                               impute_options , 
                               horizontal = True )
    
    if impute_choice == impute_options [ 0 ] : # No
        st.write ( 'Brave choice! be aware that things may break!' )
    elif impute_choice == impute_options [ 1 ] : # drop nans
        df_ip = df_ip.dropna()
    elif impute_choice == impute_options [ 2 ] : # impute mean
        df_ip = df_ip.fillna ( df_ip.mean(numeric_only=True) )
    elif impute_choice == impute_options [ 3 ] : # impute median
        df_ip = df_ip.fillna ( df_ip.median(numeric_only=True) )
    elif impute_choice == impute_options [ 4 ] : # impute mode
        df_ip = df_ip.fillna ( df_ip.mode(numeric_only=True).iloc[0] )

    df_ip['isospin_asymmetry'] = ( df_ip['n'] - df_ip['z'] ) / ( df_ip['n'] + df_ip['z'] )

    
    # I wrote an encoding routine that accomplishes the below but I used a for 
    # loop which I recall being told is slow. So I dropped it into ChatGPT 5.0
    # and asked 'can this be done better?' and it output this robust bool
    # method. (08.10.2025)
    
    parity_conditions = [
    (df_ip['n'] % 2 == 0) & (df_ip['z'] % 2 == 0), # is n&z even?
    ((df_ip['n'] % 2 != 0) & (df_ip['z'] % 2 == 0)) | # is n&z even&odd or...  
    ((df_ip['n'] % 2 == 0) & (df_ip['z'] % 2 != 0)), # vice versa?
    (df_ip['n'] % 2 != 0) & (df_ip['z'] % 2 != 0) # is n&z odd?
    ]

    parity_labels = ['Even-Even', 'Even-Odd', 'Odd-Odd']
    df_ip['parity'] = np.select(parity_conditions, parity_labels,default=str)

    exp_slider = st.slider("Choose exponent (-10 to 10):", -10., 10., 1., 1.)

    if exp_slider >= 0 :
        exp = exp_slider
    else :
        exp = 1/abs(exp_slider)

    # ChatGPT 5.0 wrote the below attempt at handling division by 0. I tried my
    # hand at it first and with no success I ask ChatGPT to try its hand at it.
    # it was as equally successful but I have left what it wrote. (10.10.2025)

    df_ip['a_trans'] = np.where(
            exp_slider < 0,
            df_ip['a'] ** (1 / abs(exp_slider)),  # negative = root
            df_ip['a'] ** exp_slider               # positive = power
    )

    if impute_choice == impute_options [ 0 ] :

        df_ip = df_ip.dropna(subset=['radius_val', 'a_trans'])

    scaling_options = [ 'None' , 'Standard' , 'Min/Max']
    scaling_string = 'Would you like to apply a Scaler to the data?'
    scaling_choice = st.radio ( scaling_string ,
                               scaling_options ,
                               horizontal = True )

    # I wrote some code to apply a scaler to the data and recombine it all for 
    # plotting and such. After, I asked chatGPT 5.0 if there was way to do it 
    # better and it generated this. (08.10.25)

    if scaling_choice != 'None':
        numeric_cols = df_ip.select_dtypes(include=["number"]).copy()

        cols_to_scale = st.multiselect(
            "Select columns to apply scaling to:",
            options=numeric_cols.columns.tolist(),
            default=['radius_val']
        )

        if scaling_choice == 'Standard' and cols_to_scale:
            scaler = StandardScaler()
            numeric_cols[cols_to_scale] = scaler.fit_transform(
                numeric_cols[cols_to_scale])
        elif scaling_choice == 'Min/Max' and cols_to_scale:
            scaler = MinMaxScaler()
            numeric_cols[cols_to_scale] = scaler.fit_transform(
                numeric_cols[cols_to_scale])

        non_numeric_cols = df_ip.select_dtypes(exclude=["number"]).copy()
        df_ip = pd.concat([non_numeric_cols, numeric_cols], axis=1)

    else:
        df_ip = df_ip.copy()

    encoder_options = [ 'None' , 'Isospin Asymmetry' , 'Parity' ]
    encoder_string = 'Select an encoder to color the  scatter plot by.'
    encoder_choice = st.radio ( encoder_string ,
                            encoder_options ,
                            horizontal = True )

    if encoder_choice == encoder_options [ 0 ] :
        hue = None
    if encoder_choice == encoder_options [ 1 ] :
        hue = 'isospin_asymmetry'
    if encoder_choice == encoder_options [ 2 ] :
        hue = 'parity'

    # # Ensure critical columns exist
    # for col in ['isospin_asymmetry', 'parity','a']:
    #     if col not in df_scaled.columns:
    #         df_scaled[col] = df_ip[col]

    X = df_ip[['radius_val']].values
    y = df_ip['a_trans'].values

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    slope = model.coef_[0]
    intercept = model.intercept_

    # these commented out lines below are the first iteration of the 
    # visualization. after making it, I dropped it into chatGPT 5.0 and asked
    # if it can be done with plotly. that code is now implemented (15.10.25)

    # Create a DataFrame with X, y, and predictions
    df_plot = pd.DataFrame({
        'radius_val': df_ip['radius_val'],
        'a_trans': df_ip['a_trans'],
        'y_pred': y_pred , 
        'a' : df_ip['a'],
        'z' : df_ip['z'],
        'n' : df_ip['n'],
        'element' : df_ip['EL'],
        'residual' : np.round(df_ip['a_trans']-y_pred,2),
        'isospin_asymmetry' : df_ip['isospin_asymmetry'],
        'parity' : df_ip['parity']
    })

    # Create scatter plot with regression line
    fig = px.scatter(
        df_plot,
        x='radius_val',
        y='a_trans',
        opacity=0.7,
        color=hue,
        labels={'radius_val': 'radius_val', 'a_trans': f'a^{exp:.3f}'},
        title=f"radius_val vs a^{exp:.3f} with Linear Fit",
        hover_data={'element': True,'a': True,'n': True,'z': True,
                    'radius_val': True, 'a_trans': True, 'y_pred': True,
                    'residual': True}
    )

    # Add regression line in red
    fit_line = px.line(df_plot, x='radius_val', y='y_pred')
    fit_line.data[0].update(line=dict(color='red', width=2))
    fig.add_traces(fit_line.data)

    # Add R^2 text annotation
    fig.add_annotation(
        x=0.05, y=0.95, xref='paper', yref='paper',
        text=f"y = {slope:.3e} x + {intercept:.3e}<br>R² = {r2:.3f}",
        showarrow=False,
        font=dict(size=22),
        align="left"
    )

    # Update all axis titles and tick labels to larger font
    fig.update_layout(
        title_font_size=20,
        xaxis=dict(title_font=dict(size=22), tickfont=dict(size=14)),
        yaxis=dict(title_font=dict(size=22), tickfont=dict(size=14)),
        legend=dict(font=dict(size=14))
    )

    # Show interactive plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # fig, ax = plt.subplots()
    # ax.scatter(df_ip['radius_val'], df_ip['a_trans'],
    #             alpha=0.7, label='Data')
    # ax.plot(df_ip['radius_val'], y_pred, color='red', label='Fit')
    # ax.grid(ls='--',alpha=0.5)


    # eq_text = f"y = {slope:.3e} x + {intercept:.3e}\n$R^2$ = {r2:.3f}"
    # ax.text(0.05, 0.95, eq_text, transform=ax.transAxes, fontsize=12,
    #          verticalalignment='top')

    # ax.set_xlabel('radius_val')
    # ax.set_ylabel(f"a^({exp:.3f})")
    # ax.tick_params(direction='in')
    # ax.legend()

    # st.pyplot(fig)

    if exp_slider == -3 :
        st.write(r'''Woah, that looks Good! So, if the radius of the nucleus is 
        well represented by the cubic root of the mass number, what does this 
        imply about the volume of a nucleus?''')

        hint = st.checkbox('Click me for a hint!')
        if hint == True :
            st.write ( '''What happens if we pretend the nucleus is a sphere?''' )
        answer = st.checkbox ('Click me for the answer.')
        if answer == True :
            st.markdown (r'''
                        If we model the nucleus as a sphere, then we can say 
                        that the volume of the nucleus is given by,

                        $V_{\text{nucleus}}=V_{\text{sphere}}=\frac{4}{3}\pi r_c^3=\frac{4}{3}\pi a^{3/3}$
                         
                        $\implies V_{\text{nucleus}} \propto a$
                         
                        That is, the nucleus is comprised mostly of the 
                        number of protons and neutrons it has!
                        ''')
            
if page == page_options[5]:  # Machine Learning Applications
    st.write("Let's try some Regression Models")

    df_ml = df_merge.copy()

    # ----------------------------
    # Imputation
    # ----------------------------
    impute_options = ['No', 'Drop NaNs', 'Impute Mean', 'Impute Median', 'Impute Mode']
    impute_choice = st.radio(
        'The dataset might be dirty, would you like to do anything about that?',
        impute_options, horizontal=True
    )

    if impute_choice == 'Drop NaNs':
        df_ml = df_ml.dropna()
    elif impute_choice == 'Impute Mean':
        df_ml = df_ml.fillna(df_ml.mean(numeric_only=True))
    elif impute_choice == 'Impute Median':
        df_ml = df_ml.fillna(df_ml.median(numeric_only=True))
    elif impute_choice == 'Impute Mode':
        df_ml = df_ml.fillna(df_ml.mode(numeric_only=True).iloc[0])

    # ----------------------------
    # Feature engineering
    # ----------------------------
    df_ml['isospin_asymmetry'] = (df_ml['n'] - df_ml['z']) / (df_ml['n'] + df_ml['z'])

    parity_conditions = [
        (df_ml['n'] % 2 == 0) & (df_ml['z'] % 2 == 0),
        ((df_ml['n'] % 2 != 0) & (df_ml['z'] % 2 == 0)) | ((df_ml['n'] % 2 == 0) & (df_ml['z'] % 2 != 0)),
        (df_ml['n'] % 2 != 0) & (df_ml['z'] % 2 != 0)
    ]
    parity_labels = ['Even-Even', 'Even-Odd', 'Odd-Odd']
    df_ml['parity'] = np.select(parity_conditions, parity_labels, default='Unknown')

    # ----------------------------
    # Scaling
    # ----------------------------
    scaling_options = ['None', 'Standard', 'Min/Max']
    scaling_choice = st.radio('Would you like to apply a Scaler to the data?', scaling_options, horizontal=True)

    numeric_cols = df_ml.select_dtypes(include=np.number).copy()
    X_columns = numeric_cols.columns.tolist()

    if scaling_choice != 'None':
        cols_to_scale = st.multiselect(
            "Select columns to scale:",
            options=X_columns,
            default=X_columns
        )
        if cols_to_scale:
            scaler = StandardScaler() if scaling_choice == 'Standard' else MinMaxScaler()
            numeric_cols[cols_to_scale] = scaler.fit_transform(numeric_cols[cols_to_scale])
    df_ml[numeric_cols.columns] = numeric_cols

    # ----------------------------
    # Select y (target)
    # ----------------------------
    y_column = st.selectbox('Select the target column (y):', options=X_columns)
    X_columns_filtered = [col for col in X_columns if col != y_column]
    plot_x_col = st.selectbox('Select feature for x-axis in prediction plot:', X_columns_filtered)

    # ----------------------------
    # Prepare X and y
    # ----------------------------
    X = df_ml[X_columns_filtered].values
    y = df_ml[y_column].values

    # Drop rows with NaNs in X or y
    mask = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]

    # ----------------------------
    # Model selection
    # ----------------------------
    model_choice = st.selectbox("Select a model to train:", ["Ridge", "Lasso"])
    alpha = st.slider("Set regularization strength (alpha):", 0.0, 10.0, 1.0, 0.1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Ridge(alpha=alpha) if model_choice == "Ridge" else Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ----------------------------
    # Metrics
    # ----------------------------
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # ----------------------------
    # Prediction plot
    # ----------------------------

    x_index = X_columns_filtered.index(plot_x_col)

    df_plot_model = pd.DataFrame({
        'x_feature': X_test[:, x_index],
        'y_actual': y_test,
        'y_pred': y_pred
    })

    fig_model = px.scatter(
        df_plot_model, x='x_feature', y='y_actual', opacity=0.7,
        title=f"{model_choice} Regression: {plot_x_col} vs {y_column}",
        labels={'x_feature': plot_x_col, 'y_actual': y_column}
    )
    fig_model.add_traces(px.line(df_plot_model, x='x_feature', y='y_pred').data)
    st.plotly_chart(fig_model, use_container_width=True)

    st.write(f"**{model_choice} Regression Results:**")
    st.write(f"R² = {r2:.3f}")
    st.write(f"MSE = {mse:.3e}")
    st.write(f"MAE = {mae:.3e}")

    # ----------------------------
    # Feature importance plot
    # ----------------------------
    feature_importance = pd.DataFrame({
        'Feature': X_columns_filtered,
        'Coefficient': model.coef_
    })
    feature_importance['Importance'] = feature_importance['Coefficient'].abs()
    feature_importance = feature_importance.sort_values(by='Importance', ascending=True)

    fig_feat = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title=f'{model_choice} Feature Importance',
        text='Coefficient'
    )
    st.plotly_chart(fig_feat, use_container_width=True)



#-------------------------------------------------------------------------------
#                                  END PROGRAM                                  
#===============================================================================