%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%															%
%		Modified from Jospeh MacMillan's undergraduate template:	%
%		http://faculty.uoit.ca/macmillan/thesis_template.zip			%
%															%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{UOIT_Thesis}[2018/12/15 Brady Metherall's UOIT Thesis Template]

\DeclareOption{BW}{\def\@bw{}}
\DeclareOption*{%
  \PassOptionsToClass{\CurrentOption}{report}%
}
\ProcessOptions\relax
\LoadClass{report}

% Math
\RequirePackage{amsmath}
\RequirePackage{amssymb}
% Graphics
\RequirePackage{graphicx}
\RequirePackage[dvipsnames]{xcolor}
% Margins and spacing
\RequirePackage[top=1in, bottom=1in, left=1.5in, right=1in]{geometry}
\RequirePackage{setspace}
% Formatting
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{datetime}
\RequirePackage{anyfontsize}
% Other
\RequirePackage{hyperref}
\RequirePackage{subcaption}

% Define macros
\newdateformat{monthyeardate}{\monthname[\THEMONTH] \THEYEAR}
\newcommand\chapnumsize{\fontsize{60}{25}\selectfont}
\ifdefined\@bw
  \definecolor{uoitblue}{RGB}{0,0,0}
\else
  \definecolor{uoitblue}{RGB}{0,60,113}
\fi

% Define bibliography style
\bibliographystyle{ieeetr}
\renewcommand{\bibname}{References}

% Define headers / footers
\pagestyle{fancy}
\renewcommand{\chaptermark}[1]{\markboth{\chaptername \ \thechapter: #1}{}}
\lhead{\small \leftmark}
\chead{}
\rhead{\small \thepage}
\lfoot{}
\rfoot{}
\cfoot{}

% Chapter header style
\titleformat{\chapter}[display]
{\normalfont\bfseries}
{\color{uoitblue}\small\chaptername \ \chapnumsize\thechapter\filleft}
{10ex}
{\titlerule[0.5pt]
\vspace{2ex}%
\color{uoitblue}\LARGE\filcenter}
[\vspace{1ex}%
{\color{black}\titlerule[2pt]}]

% Title page style
\renewcommand{\maketitle}
{
\newgeometry{top=1in, bottom=1in, left=2.5in, right=2in}
\begin{titlepage}
\begin{center}
{
\color{uoitblue}\Huge \textbf{\@title}\\
\vspace{8mm}
\color{black}
\normalsize by \\
\vspace{2mm}
\Large \@author \\
\vspace{25mm}

\normalsize

A Thesis Submitted in Partial Fulfillment \\
of the Requirements for the Degree of \\
Master of Science \\
in \\
The Faculty of Science \\
Modelling and Computational Science \\
\vspace{5mm}
University of Ontario Institute of Technology \\
\vspace{5mm}
\monthyeardate\today
\vspace{10mm}

\ifdefined\@bw
  \includegraphics[scale=0.5]{Coat_of_Arms_BW}
\else
  \includegraphics[scale=0.5]{Coat_of_Arms}
\fi

\vspace{10mm}

\textcopyright \ \@author \ \the\year
}
\end{center}
\end{titlepage}
\restoregeometry
\setcounter{page}{2}
}

% Abstract
\newcommand{\makeabstract}[1]
{
\clearpage
\phantomsection
\thispagestyle{plain}
\addcontentsline{toc}{chapter}{\numberline{}Abstract}
\vspace{5pt}
\begin{center} \bf{\Large{Abstract}} \end{center}
\vspace{20pt}
#1
\clearpage
}

% Acknowledgements / dedication
\newcommand{\makeacknowledgements}[1]
{
\clearpage
\phantomsection
\thispagestyle{plain}
\addcontentsline{toc}{chapter}{\numberline{}Acknowledgements}
\vspace{5pt}
\begin{center} \bf{\Large{Acknowledgements}} \end{center}
\vspace{20pt}
#1
\clearpage
}

% Declaration
\newcommand{\makedeclaration}
{
\clearpage
\phantomsection
\thispagestyle{plain}
\addcontentsline{toc}{chapter}{\numberline{}Author's Declaration}
\vspace{5pt}
\begin{center} \bf{\Large{Author's Declaration}} \end{center}
\vspace{20pt}

\noindent I declare that the work in this thesis was carried out in accordance with the regulations of the University of Ontario Institute of Technology. The work is original except where indicated by special reference in the text and no part of the dissertation has been submitted for any other degree. Any views expressed in the dissertation are those of the author and in no way represent those of the University of Ontario Institute of Technology. The thesis has not been presented to any other university for examination either in Canada or elsewhere.

\vspace{25mm}
\begin{center}
\rule{0.8\textwidth}{0.5pt} \\
\@author \\
\@date
\end{center}
\clearpage
}

% Table of Contents
\newcommand{\maketableofcontents}
{
\clearpage
\phantomsection

\addcontentsline{toc}{chapter}{\numberline{}Table of Contents}
\tableofcontents

\newpage
\phantomsection
\addcontentsline{toc}{chapter}{\numberline{}List of Figures}
\listoffigures

\newpage
\phantomsection
\addcontentsline{toc}{chapter}{\numberline{}List of Tables}
\listoftables

\clearpage
}

\endinput
