%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  
%%  UOIT_Thesis.cls (v1.0)
%%
%%  This program can be redistributed and/or modified under the terms
%%  of the LaTeX Project Public License Distributed from CTAN archives
%%  in directory CTAN:/macros/latex/base/lppl.txt.
%%
%%  Copyright (c) 2018 by Brady Metherall
%%  Last Update: 2018 December 20
%%  
%%  
%%  This file is distributed in the hope that it will be useful but
%%  without any warranty.
%%  
%%  Feel free to copy and redistribute this file, as long as this
%%  copyright notice remains intact. You may also modify the file
%%  to suit your particular needs, as long as you indicate your
%%  changes below.
%%  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%  REVISION HISTORY:
%% 
%%  v1.1: 2019/09/24 (Brady Metherall)
%%      - Renamed to new branding
%% 
%%  v1.1: 2019/07/15 (Brady Metherall)
%%      - Added Thesis Examination Information and Statement of Contributions
%% 
%%  v1.0: 2018/12/20 (Brady Metherall)
%%      - Repurposed jthesis.cls file for UOIT graduate theses.
%%
%%
%%  v1.0: 2014/08/20 (Joseph MacMillan)
%%      - Initial version
%%      - http://faculty.uoit.ca/macmillan/thesis_template.zip
%%  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{Ontario_Tech_Thesis}[2019/09/24 Brady Metherall's Ontario Tech Thesis Template]

% Optional arguments from the document
\DeclareOption{BW}{\def\@bw{}} % Black and white
\DeclareOption{nocoat}{\def\@nocoat{}} % No coat of arms
\DeclareOption{phd}{\def\@phd{}} % PhD
\DeclareOption{oneside}{ % One sided document
  \def\@oneside{}
  \PassOptionsToClass{\CurrentOption}{book}
}
\DeclareOption*{
  \PassOptionsToClass{\CurrentOption}{book}%
}
\ProcessOptions\relax
\LoadClass[12pt, letterpaper]{book}

% Math
\RequirePackage{amsmath}
\RequirePackage{amssymb}
% Graphics
\RequirePackage{graphicx}
\RequirePackage[dvipsnames]{xcolor}
% Margins and spacing
\RequirePackage{geometry}
\RequirePackage{setspace}
% Formatting
\RequirePackage{fancyhdr}
\RequirePackage{titlesec}
\RequirePackage{datetime}
\RequirePackage{anyfontsize}
\RequirePackage{emptypage}
% Other
\RequirePackage[colorlinks=true, linkcolor=uoitblue, urlcolor = uoitblue]{hyperref}
\RequirePackage{subcaption}
\RequirePackage{cite}

% Define some things
\newdateformat{monthyeardate}{\monthname[\THEMONTH] \THEYEAR} % New date format
\newcommand\chapnumsize{\fontsize{75}{0}\selectfont} % Chapter number font size
\ifdefined\@bw
  \definecolor{uoitblue}{RGB}{0,0,0}
\else
  \definecolor{uoitblue}{RGB}{0,60,113} % Colour of chapter titles (UOIT's dark blue)
\fi

% Get info from main tex document
\newcommand*{\degree}[1]{\def\@degree{#1}}
\newcommand*{\faculty}[1]{\def\@faculty{#1}}
\newcommand*{\program}[1]{\def\@program{#1}}
\newcommand*{\defense}[1]{\def\@defense{#1}}
\newcommand*{\university}[1]{\def\@university{#1}}
\newcommand*{\city}[1]{\def\@city{#1}}
\university{University of Ontario Institute of Technology} % Default university if none given
\city{Oshawa, Ontario, Canada} % Default city if none given

\newcommand*{\chair}[1]{\def\@chair{#1}}
\newcommand*{\supervisor}[1]{\def\@supervisor{#1}}
\newcommand*{\examiner}[1]{\def\@examiner{#1}}
\newcommand*{\univexaminer}[1]{\def\@univexaminer{#1}}
\newcommand*{\committee}[2]{\def\@committeea{#1}\def\@committeeb{#2}}

% PDF metadata
\AtBeginDocument{
  \hypersetup{
    pdftitle=\@title,
    pdfauthor=\@author,
    pdfsubject=\@degree---\@university
  }
}

% Define margins
\ifdefined\@oneside
  \geometry{top=1in, bottom=1in, left=1.5in, right=1in}
\else
  \geometry{top=1in, bottom=1in, outer=1.5in, inner=1in}
\fi

% Define bibliography style
\bibliographystyle{ieeetr}
\renewcommand{\bibname}{References}

% Define headers / footers
\pagestyle{fancy}
\renewcommand{\chaptermark}[1]{\markboth{\chaptername \ \thechapter: #1}{}}
\renewcommand{\sectionmark}[1]{\markright{Section \thesection: #1}{}}
\fancyhead[RO, LE]{\small \thepage} % Outside of page
\fancyhead[RE]{\small \nouppercase{\leftmark}} % Inside of left page
\ifdefined\@oneside
  \fancyhead[LO]{\small \nouppercase{\leftmark}} % Inside of right page
\else
  \fancyhead[LO]{\small \nouppercase{\rightmark}} % Inside of right page
\fi
\fancyfoot{}

% Chapter header style
\titleformat{\chapter}[display]
{\normalfont\bfseries}
{\color{uoitblue}\small\chaptername \ \chapnumsize\thechapter\filleft}
{5mm}
{\titlerule[0.5pt]
\vspace{2mm}%
\color{uoitblue}\LARGE\filcenter}
[\vspace{2mm}%
{\color{black}
\titlerule[2pt]}]

% Title page style
\renewcommand{\maketitle}
{
\begin{titlepage}
\begin{center}
{
\color{uoitblue}\Huge \textbf{\@title}\\
\vfill
\color{black}
\normalsize by \\
\vfill
\Large \@author \\
\vfill \vfill

\normalsize
A Thesis Submitted in Partial Fulfillment \\
of the Requirements for the Degree of \\
\@degree \\
in \\
The Faculty of \@faculty \\
\@program \\
\vfill \vfill
\@university \\
\@city \\
\vfill \vfill
\monthyeardate\today

\ifdefined\@nocoat
\else
  \ifdefined\@bw
    \vfill \vfill
    \includegraphics[scale=0.5]{Coat_of_Arms_BW}
  \else
    \vfill \vfill
    \includegraphics[scale=0.5]{Coat_of_Arms}
  \fi
\fi

\vfill \vfill

\textcopyright \ \@author \ \the\year
}
\end{center}
\end{titlepage}
\setcounter{page}{2}
}

% Thesis Examination Information
\newcommand{\makeexaminfo}{
\clearpage
\phantomsection
\thispagestyle{plain}
\addcontentsline{toc}{chapter}{\numberline{}Thesis Examination Information}
\vfill
\begin{center}
\textbf{\Large{Thesis Examination Information}} \\
\vfill
Submitted by: \@author \\
\vfill
\@degree \ in \@program \\
\vfill
\@title \\
\end{center}
\vfill \vfill
An oral defense of this thesis took place on \@defense \ in front of the following examining committee: \\
\vfill
\noindent \textbf{Examining Committee:} \\
\renewcommand{\arraystretch}{2}
\begin{tabular}{@{} p{6cm} p{10cm}}
Chair of Examining Committee & \@chair \\
Research Supervisor & \@supervisor \\
Examining Committee Member & \@committeea \\
Examining Committee Member & \@committeeb \\
\ifdefined\@phd University Examiner & \@univexaminer \\ \fi
\ifdefined\@phd External \else Thesis \fi Examiner & \@examiner
\end{tabular}
\renewcommand{\arraystretch}{1}
\vfill \vfill
\noindent The above committee determined that the thesis is acceptable in form and content and that a satisfactory knowledge of the field covered by the thesis was demonstrated by the candidate during an oral examination. A signed copy of the Certificate of Approval is available from the School of Graduate and Postdoctoral Studies.
\vfill \vfill \vfill \vfill \vfill \vfill
\clearpage
}

% Abstract
\newcommand{\makeabstract}[1]{
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
\newcommand{\makeacknowledgements}[1]{
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
\newcommand{\makedeclaration}{
\clearpage
\phantomsection
\thispagestyle{plain}
\addcontentsline{toc}{chapter}{\numberline{}Author's Declaration}
\vspace{5pt}
\begin{center} \bf{\Large{Author's Declaration}} \end{center}
\vspace{20pt}

\noindent I declare that the work in this thesis was carried out in accordance with the regulations of the \@university. The work is original except where indicated by special reference in the text and no part of the dissertation has been submitted for any other degree. Any views expressed in the dissertation are those of the author and in no way represent those of the \@university. This thesis has not been presented to any other university for examination either in Canada or elsewhere.

\vspace{25mm}
\begin{center}
\rule{0.8\textwidth}{0.5pt} \\
\@author \\
\@date
\end{center}
\clearpage
}

% Statement of Contributions
\newcommand{\makecontributions}[1]{
\clearpage
\phantomsection
\thispagestyle{plain}
\addcontentsline{toc}{chapter}{\numberline{}Statement of Contributions}
\vspace{5pt}
\begin{center} \bf{\Large{Statement of Contributions}} \end{center}
\vspace{20pt}
#1
\clearpage
}

% Table of Contents
\newcommand{\maketableofcontents}{
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

% Change chapter header to 'Appendix' for appendices
\makeatletter
\g@addto@macro{\appendix}{\renewcommand\chaptername{Appendix}}
\makeatother

\endinput