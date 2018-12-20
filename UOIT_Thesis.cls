%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%															%
%		Modified from Jospeh MacMillan's undergraduate template:	%
%		http://faculty.uoit.ca/macmillan/thesis_template.zip			%
%															%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{UOIT_Thesis}[2018/12/15 Brady Metherall's UOIT Thesis Template]

% Optional arguments from the document
\DeclareOption{BW}{\def\@bw{}} % Black and white
\DeclareOption{nocoat}{\def\@nocoat{}} % No coat of arms
\DeclareOption{oneside}{ % One sided document
  \def\@oneside{}
  \PassOptionsToClass{\CurrentOption}{book}
}
\DeclareOption*{
  \PassOptionsToClass{\CurrentOption}{book}%
}
\ProcessOptions\relax
\LoadClass[12pt, letter]{book}

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
\RequirePackage[colorlinks=true, linkcolor=uoitblue]{hyperref}
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
\newcommand*{\university}[1]{\def\@university{#1}}
\university{University of Ontario Institute of Technology} % Default university if none given

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
\vfill \vfill
\monthyeardate\today

\ifdefined\@nocoat
\else
  \ifdefined\@bw
    \vfill \vfill
    \includegraphics[scale=0.6]{Coat_of_Arms_BW}
  \else
    \vfill \vfill
    \includegraphics[scale=0.6]{Coat_of_Arms}
  \fi
\fi

\vfill \vfill

\textcopyright \ \@author \ \the\year
}
\end{center}
\end{titlepage}
\setcounter{page}{2}
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