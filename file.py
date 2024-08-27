text = """
% This must be in the first 5 lines to tell arXiv to use pdfLaTeX, which is strongly recommended.
\pdfoutput=1
% In particular, the hyperref package requires pdfLaTeX in order to break URLs across lines.

\documentclass[11pt]{article}

% Change "review" to "final" to generate the final (sometimes called camera-ready) version.
% Change to "preprint" to generate a non-anonymous version with page numbers.
\usepackage[review]{coling}

% Standard package includes
\usepackage{times}
\usepackage{latexsym}

% For proper rendering and hyphenation of texts containing Latin characters (including in bib files)
\usepackage[T1]{fontenc}
% For Vietnamese characters
% \usepackage[T5]{fontenc}
% See https://www.latex-project.org/help/documentation/encguide.pdf for other character sets

% This assumes your files are encoded as UTF8
\usepackage[utf8]{inputenc}

% This is not strictly necessary, and may be commented out,
% but it will improve the layout of the manuscript,
% and will typically save some space.
\usepackage{microtype}

% This is also not strictly necessary, and may be commented out.
% However, it will improve the aesthetics of text in
% the typewriter font.
\usepackage{inconsolata}

%Including images in your LaTeX document requires adding
%additional package(s)
\usepackage{graphicx}

\usepackage{booktabs}
\usepackage{multirow}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{float} % 加载 float 包以使用 [H] 选项
\usepackage{ulem}
\usepackage{algorithm} 
% \usepackage{algorithmic} 
\usepackage{algorithmicx}
\usepackage{algpseudocode} 
\usepackage{enumitem}

% -------------------------允许算法跨页-------------
\makeatletter
\newenvironment{breakablealgorithm}
 {% \begin{breakablealgorithm}
  \begin{center}
   \refstepcounter{algorithm}% New algorithm
   \hrule height.8pt depth0pt \kern2pt% \@fs@pre for \@fs@ruled
   \renewcommand{\caption}[2][\relax]{% Make a new \caption
    {\raggedright\textbf{\ALG@name~\thealgorithm} ##2\par}%
    \ifx\relax##1\relax % #1 is \relax
     \addcontentsline{loa}{algorithm}{\protect\numberline{\thealgorithm}##2}%
    \else % #1 is not \relax
     \addcontentsline{loa}{algorithm}{\protect\numberline{\thealgorithm}##1}%
    \fi
    \kern2pt\hrule\kern2pt
   }
 }{% \end{breakablealgorithm}
   \kern2pt\hrule\relax% \@fs@post for \@fs@ruled
  \end{center}
 }
\makeatother



% If the title and author information does not fit in the area allocated, uncomment the following
%
%\setlength\titlebox{<dim>}
%
% and set <dim> to something 5cm or larger.

\title{GSIFN: A Graph-Structured and Interlaced-Masked Multimodal Transformer Based Fusion Network for Multimodal Sentiment Analysis}

% Author information can be set in various styles:
% For several authors from the same institution:
% \author{Author 1 \and ... \and Author n \\
%     Address line \\ ... \\ Address line}
% if the names do not fit well on one line use
%     Author 1 \\ {\bf Author 2} \\ ... \\ {\bf Author n} \\
% For authors from different institutions:
% \author{Author 1 \\ Address line \\ ... \\ Address line
%     \And ... \And
%     Author n \\ Address line \\ ... \\ Address line}
% To start a separate ``row'' of authors use \AND, as in
% \author{Author 1 \\ Address line \\ ... \\ Address line
%     \AND
%     Author 2 \\ Address line \\ ... \\ Address line \And
%     Author 3 \\ Address line \\ ... \\ Address line}

\author{First Author \\
 Affiliation / Address line 1 \\
 Affiliation / Address line 2 \\
 Affiliation / Address line 3 \\
 \texttt{email@domain} \\\And
 Second Author \\
 Affiliation / Address line 1 \\
 Affiliation / Address line 2 \\
 Affiliation / Address line 3 \\
 \texttt{email@domain} \\}

%\author{
% \textbf{First Author\textsuperscript{1}},
% \textbf{Second Author\textsuperscript{1,2}},
% \textbf{Third T. Author\textsuperscript{1}},
% \textbf{Fourth Author\textsuperscript{1}},
%\\
% \textbf{Fifth Author\textsuperscript{1,2}},
% \textbf{Sixth Author\textsuperscript{1}},
% \textbf{Seventh Author\textsuperscript{1}},
% \textbf{Eighth Author \textsuperscript{1,2,3,4}},
%\\
% \textbf{Ninth Author\textsuperscript{1}},
% \textbf{Tenth Author\textsuperscript{1}},
% \textbf{Eleventh E. Author\textsuperscript{1,2,3,4,5}},
% \textbf{Twelfth Author\textsuperscript{1}},
%\\
% \textbf{Thirteenth Author\textsuperscript{3}},
% \textbf{Fourteenth F. Author\textsuperscript{2,4}},
% \textbf{Fifteenth Author\textsuperscript{1}},
% \textbf{Sixteenth Author\textsuperscript{1}},
%\\
% \textbf{Seventeenth S. Author\textsuperscript{4,5}},
% \textbf{Eighteenth Author\textsuperscript{3,4}},
% \textbf{Nineteenth N. Author\textsuperscript{2,5}},
% \textbf{Twentieth Author\textsuperscript{1}}
%\\
%\\
% \textsuperscript{1}Affiliation 1,
% \textsuperscript{2}Affiliation 2,
% \textsuperscript{3}Affiliation 3,
% \textsuperscript{4}Affiliation 4,
% \textsuperscript{5}Affiliation 5
%\\
% \small{
%  \textbf{Correspondence:} \href{mailto:email@domain}{email@domain}
% }
%}

\begin{document}
\maketitle


\begin{abstract}

Multimodal Sentiment Analysis (MSA) focuses on leveraging various signals to interpret human emotions, typically tackled by advancing fusion techniques and representation learning. Our proposed GSIFN addresses key challenges in modal interaction efficiency and high computational costs associated with existing large models for non-text feature extraction. GSIFN incorporates two main components: (i) a Graph-structured and Interlaced-Masked Multimodal Transformer that employs an Interlaced Mask mechanism for robust multimodal Graph embedding and Transformer-based modal interaction, and (ii) a Self-supervised learning framework with a non-text modal feature enhancement module. This framework ensures efficient integration of multimodal data while reducing computational load. Additionally, it utilizes a parallelized LSTM with matrix memory to bolster the non-verbal modal unimodal label generation process. Evaluated on the MSA datasets CMU-MOSI, CMU-MOSEI, and CH-SIMS, GSIFN demonstrates superior performance with significantly lower computational costs compared to state-of-the-art methods.


\end{abstract}

\section{Introduction}
Multimodal deep learning integrates different forms of data for sentiment analysis to achieve more natural human-computer interactions. With the rise of social media, where users express sentiment through text, video, and audio, multimodal sentiment analysis (MSA) has become a popular research area \cite{eswa:fmlmsn, applintelli:hfngc, ipm:cmhfm}. MSA typically relies on text, vision, and audio modalities for sentiment polarity prediction, its main challenges include integrating inconsistent sentiment information and semantic alignment across these modalities. Methods involve designing effective fusion strategies \citep{emnlp:tfn, acl:mult, emnlp:almt} to integrate heterogeneous data for comprehensive emotional representation and semantic alignment, as well as developing representation learning strategies \citep{aaai:self-mm, acl:confede, taffc:mtmd} to enhance single-modal information and model robustness.

 % Earlier models like TFN and MFN focus on modality fusion but lacked sufficient temporal information utilization and effective alignment strategies. Transformer-based models, such as MulT, TETFN, and ALMT, introduced cross-modal attention mechanisms and text dominant fusion enhancement to better handle modality temporal information and improve fusion and alignment. To further enhance robustness and improve the consistency and representation of modality information, models like Self-MM and ConFEDE employ self-supervised or contrastive learning were proposed.

Despite achieving some successes, these approaches still face three main challenges. First, for the models which focus on modal fusion, due to the widespread use of cross-modal attention mechanism and the decoupling of the combination of different modalities to be fused into multiple independent modules, this kind of scheme makes the model fail to fully integrate the representation information of the three major modalities, instead retaining redundancy in unimodal information. This leads to the model weights being excessively redundant and in need of pruning, and once the naive serial weight sharing strategy is applied \citep{mm:misa} to share the representation information of each modality, information disorder occurs. Second, for the representation learning-based models, the data extraction and representation module of non-text modalities cannot effectively balance the amount of parameters and representation efficiency. Small models (GRU, LSTM, etc.) or conventional extractors (OpenFace, LibROSA, etc.) cause excessive loss of representation of non-verbal modals, while large models (ViT, Wav2Vec, etc.) incur excessive overhead. Third, models employ both of the aforementioned approaches face both of these drawbacks, thus it is significant to weigh the pros and cons.

To address the aforementioned issues, we propose a model called \textbf{G}raph-\textbf{S}tructured and \textbf{I}nterlaced-Masked Multimodal Transformer Based \textbf{F}usion \textbf{N}etwork, dubbed \textbf{GSIFN}. There are two attractive properties in GSIFN. First, in the process of multimodal fusion, it realizes efficient and low overhead representation information sharing without information disorder. To attain this, we propose Graph-structured and interlaced-masked multimodal Transformer (GsiT), which is structured in units of modal subgraphs. GsiT utilizes the Interlaced Mask (IM) mechanism to construct multimodal graph embeddings, in which Interlaced-Inter-Fusion Mask (IFM) constructs fusion graph embeddings. Interlaced-Intra-Enhancement Mask (IEM) constructs enhancement graph embedding. Specifically, with shared information, IFM constructs two opposite unidirectional ring graph embeddings to realize a complete fusion procedure, IEM constructs an internal enhancement embedding of modal subgraphs to realize the multimodal fusion enhancement. IM utilizes weight sharing strategy to realize a complete interaction of multimodal information in the fusion procedure, while eliminating useless information, thereby improving the fusion efficiency and achieving pruning. Second, it significantly reduces computing overhead brought by non-verbal modal feature enhancement operations while ensuring the robustness of the model. We employ a unimodal label generation module (ULGM) to enhance the model robustness and apply Extended Long Short Term Memory with Matrix Memory (mLSTM) to enhance non-verbal modal feature in ULGM. mLSTM is fully parallelized and has superior memory mechanism over LSTM, which can deeply mine the semantic information of non-verbal modalities. Additionally, using mLSTM could avoid the huge computational overhead caused by large models. Overall, our contributions are as follows:

\begin{itemize}[itemsep=0pt,topsep=-4pt,parsep=0pt]
  % \setlength{\topsep}{0pt} % 减少itemize环境与上方段落之间的距离
  % \setlength\itemsep{0em} % 调整项目之间的间距
  % \setlength\parskip{0em} % 调整段落之间的间距
  \item We propose GSIFN, a graph-structured and interlaced-masked multimodal transformer network. Experiments and ablation studies across various datasets validate its effectiveness and superiority.
  \item We propose GsiT, a graph-structured and interlaced-masked multimodal transformer that uses the Interlaced Mask mechanism to build multimodal graph embeddings from modal subgraphs. It ensures efficient, low-overhead information sharing, reduces spatio-temporal redundancy and noise, and yields a more compact and informative multimodal representation while lowering the module's parameter count.
  \item We use mLSTM, an extended LSTM with matrix memory, to enhance non-verbal modal features utilized for unimodal label generation. This approach improves model robustness and representation while avoiding the overhead of large models.
\end{itemize}

\begin{figure*}[htpb]
\begin{center}
  \includegraphics[width=\linewidth]{fig/arch_8.png}
\end{center}
  \caption{GSIFN Architecture.}
\label{fig:overview}
\end{figure*}

% These instructions are for authors submitting papers to the COLING 2025 conference using \LaTeX. They are not self-contained. All authors must follow the general instructions for COLING 2025 proceedings which are an adaptation of (or rely on) the general instructions for ACL proceedings\footnote{\url{http://acl-org.github.io/ACLPUB/formatting.html}}, and this document contains additional instructions for the \LaTeX{} style files.

% The templates include the \LaTeX{} source of this document (\texttt{coling\_latex.tex}),
% the \LaTeX{} style file used to format it (\texttt{coling.sty}),
% a COLING bibliography style (\texttt{coling\_natbib.bst}),
% an example bibliography (\texttt{custom.bib}),
% and the bibliography for the ACL Anthology (\texttt{anthology.bib}).

\section{Related Work}

\subsection{Multimodal Sentiment Analysis}

Multimodal sentiment analysis (MSA) is an increasingly popular research field. It mainly focuses on modals text, vision and audio. Earlier models focus on modal fusion. Zadeh et al. were among the first promoters in this field. They proposed TFN \cite{emnlp:tfn}, built a power set form of modal combination, and realized complete modal fusion by using Cartesian product, but did not consider the temporal information of non-verbal modals. Then MFN \cite{aaai:mfn} uses LSTM to extract the timing information of the three modes through explicit modal alignment (CTC, padding etc.), and uses the attention mechanism and gated memory to realize the efficient fusion of multi-modal temporal information.

With the rise of Transformer, MulT \cite{acl:mult} proposes cross-modal attention mechanism from the perspective of modal translation, which can effectively integrate multi-modal data while realizing implicit modal alignment. Based on MulT, models such as TETFN \cite{pr:tetfn} and ALMT \cite{emnlp:almt} focus on text data with stronger emotional information to enhance non-verbal modal data, thus achieving better representation and better modal fusion. MAG-BERT \cite{acl:mag-bert} uses Multimodal Adaptation Gate (MAG) to fine-tune BERT using multi-modal data. CENet \cite{tmm:cenet} constructs non-verbal modal vocabularies, realizes non-verbal modal representation enhancement, and realizes MSA capability enhancement of fine-tuned BERT.

In order to improve the robustness of the model and the representation ability of non-text modes, and thus improve the overall multi-modal emotion analysis ability of the model, representation learning-based models such as Self-MM \cite{aaai:self-mm}, ConFEDE \cite{acl:confede}, and MTMD \cite{taffc:mtmd} were proposed. They use self-supervised learning, contrast learning or knowledge distillation to achieve robust representation of modal information consistency and difference.

TETFN, MMML \cite{naacl:mmml} and AcFormer \cite{mm:acformer} combine multimodal Transformer with representation learning to effectively improve model performance, and verify the feasibility of combining the two to learn from other strengths.

Due to the excessive use of traditional multimodal Transformer architecture in these methods, these models often bring a high number of parameters in the core fusion module, and because different fusion combinations are decouped to multiple independent Transformers, the interaction of modal information is insufficient, and there are problems of insufficient weight regularity. 
In the concrete implementation, we refer to the idea of graph neural network, construct a graph-structured Transformer with modal subgraph units. 

\subsection{Graph Neural Networks}

Graph neural networks (GNN) are used in \cite{ijcnn:gnn, tnn:gnn}, attempts to extend deep neural networks to process graph-structured data. Since then, there has been growing research interest in generalized operations of deep neural networks, such as convolutions \cite{eswc:gcn, iclr:gcn, nips:gcn}, recursion \cite{nips:grn}, and attention \cite{iclr:gat, iclr:gatv2}.

MTAG \cite{naacl:mtag} uses graph attention to achieve a much lower number of parameters than MulT while maintaining the fusion effect. However, since existing datasets are originally unstructured, which makes it hard to construct multimodal graph. What is more, graph neural networks are not robust enough to face adversarial attacks \cite{nips:rob_gnn}, and paying too much attention to a certain node results in the weight of a single node always being the largest \cite{iclr:gatv2}. 

Therefore, we only consider the idea of reference graph neural network to structure Tranformer, which is then called GsiT. Specifically, we use IM mechanism to ensure the robustness of model representation and realize weight sharing, parameter reduction and multi-modal interaction without information disorder.

\subsection{Linear Attention Networks}

In the field of natural language processing, reducing the computational cost of Transformers while maintaining performance has become a popular research topic. RWKV \citep{emnlp:rwkv}, RetNet \citep{arxiv:RetNet}, Mamba \citep{arxiv:mamba}, Mamba-2 \citep{icml:mamba2} are representatives among them. xLSTM \citep{icml:xlstm}, as an extension of LSTM, introduces exponential gating to solve the limitations of memory capacity and parallelization, especially when dealing with long sequences. 

\begin{figure*}[htpb]
\begin{center}
  \includegraphics[width=\linewidth]{fig/GsiT_IM.png}
\end{center}
  \caption{GsiT Architecture and IM Mechanism.}
\label{fig:gsit_im}
\end{figure*}

At the same time, recent work in the field of MSA has begun to use more advanced feature extractors to enhance non-text modal features, taking into account the phenomenon that non-text modal representation is weak. For example, TETFN, AcFormer uses Vision Transformer (ViT) \cite{iclr:vit} to extract visual features, AcFormer uses Wav2Vec \cite{interspeech:wav2vec} to extract features, and MMML uses raw audio data to fine-tune Data2Vec \cite{icml:data2vec}. However, these methods often result in a large number of parameters, but the actual improvement over traditional features is not obvious. In order to reduce model parameters and ensure model performance at the same time, self-supervised learning method is used to strengthen the capture and characterization of information in the modes, and feature enhancement is carried out for self-supervised learning. mLSTM module in xLSTM is used to significantly reduce the computational cost compared with the model that adopts large model for feature extraction and enhancement, while ensuring model performance.

\section{Methodology}

% In this section, we introduce the task setup of multimodal sentiment analysis(Section \ref{sec:task_setup}) and the overall architecture of GSIFN(Section \ref{sec:overall_architecture}). For the proposed model, the feature encoding procedure of raw modality is first described(Section \ref{sec:modality encoding}). Then, we introduce in detail a module called Graph-Structured Cross-Modal Transformer designed to achieve modality fusion operations(Section \ref{sec:GSIFN}). Finally, we describe briefly the self-supervision learning framework our model based on(Section \ref{sec:self-supervision}).

\subsection{Preliminaries}
\label{sec:preliminaries}
The objective of multimodal sentiment analysis (MSA) is to evaluate emotion polarity using multimodal data. Existing MSA datasets generally contain three modes: $t, v, a$ represent modalities text, vision, and audio, respectively. Specially, $m$ denotes multimodality. The input of MSA task is $S_u \in \mathbb{R}^{T^s_u \times d^s_u}$. Where $u\in\{t,v,a\}$, $T^s_u$ denotes the raw sequence length and $d^s_u$ denotes the raw representation dimension of modality $u$. In this paper, we define multiple outputs $\hat{y}_u \in R$. Where $u \in \{t, v, a, m \}$, $\hat{y}_{\{t,v,a\}}$ denote unimodal outputs, obtained for unimodal label generation. $\hat{y}_m$ denotes fusion output, obtained for final prediction. Other symbols are defined as follows, fusion module input are $\{X_t,X_v,X_a\}$, since the ULGM input are $\{\mathcal{X}_t,\mathcal{X}_v,\mathcal{X}_a\}$. In particular, in the interpretation of GsiT $\{X_t, X_v X_a \}$ are abstracted to sequences of vertices $\{\mathcal{V}_t, \mathcal{V}_v, \mathcal{V}_a\}$. Labels for $y_u \in R $, where $u\in\{t,v,a,m\}$, $y_{\{t,v,a\}}$ are unimodal label generated by ULGM, $y_m$ is the ground truth label for fusion output.


\subsection{Overall Architecture}
\label{sec:overall_architecture}
The overview of our model is shown in Figure \ref{fig:overview} which consists of three major parts: (1) \textit{Modality Encoding} utilizes tokenizer (for text modality), feature extractors and temporal enhancers (firmware for non-verbal modality vision and audio) to convert raw multimodal signals into numerical feature sequences(text, vision, and audio embedding). Enhanced non-verbal modality feature is utilized for unimodal label generation in self-supervised learning framework. (2) \textit{Graph-Structured Multimodal Fusion} takes the processed text, vision and audio embedding as input. The module Graph-Structured and Interlaced-Masked Multimodal Transformer utilizes interlaced mask to construct multimodal graph embeddings. It uses weight sharing to fully interact multimodal information, meanwhile eliminating useless data and improving fusion efficiency and achieving model pruning.(3) \textit{Self-Supervised Learning Framework} generates final representations and defines positive and negative centers by projecting original text embedding, enhanced vision and audio embedding, and fused output to hidden states, whereas unimodal labels are seperately generated using text, vision, and audio representations.


\subsection{Modality Encoding}
\label{sec:modality encoding}
For text modality, we use the pretrained transfomer BERT as the text encoder. Input text token sequence is constructed by the raw sentence $S_t$ = $\{ w_1, w_2, \dots, w_n\}$ concatenated with two special tokens ([CLS] at the head and [SEP] at the end) which forms $S_t^{'}$ = $\{ [CLS],w_1, w_2, \dots, w_n,[SEP]\}$. Then, $S_t^{'}$ is inputted into the embedding layer of BERT which outputs the embedding sequence $\mathcal{X}_t$ = $\{ t_0, t_1, \dots, t_{n+1} \}$ . Following previous works, input sequences $X_u \in \mathbb{R}^{T_u \times d_u}$, where $u \in \{t,v,a\}$, $T_u$ denotes the extracted sequence length and $d_u$ denotes the extracted representation dimension of modality $u$, is extracted by one dimensional 
convolution layer from $\mathcal{X}_t$ and raw vision, audio sequences $S_{\{v,a\}}$.

% \vspace{-0.6cm} 
\begin{align}
  &X_t = \text{Conv1D}(\mathcal{X}_t) \\
  &X_{\{v,a\}} = \text{Conv1D}(S_{\{v,a\}})
\end{align}
% \vspace{-0.6cm} 

After that, we use an extended Long Short Term Memory which is fully parallelizable with a matrix memory and a covariance update rule (mLSTM) as the temporal signal enhancer of vision and audio sequence. The detailed definition of mLSTM is in Appendix \ref{apdx:xlstm}.

We use mLSTM networks to capture and enhance the temporal features of vision and audio:

\begin{equation}
  \mathcal{X}_{\{v,a\}}=\text{mLSTM}(X_{\{v,a\}})
\end{equation}

mLSTM can enhance non-verbal modal features utilized for unimodal label generation and approach improves model robustness and representation while avoiding the overhead of large models.

% \textcolor{red}{The inputs of multimodal fusion layer are $\{X_t,X_v,X_a\}$, while which of the unimodal generator are $\{\mathcal{X}_t,\mathcal{X}_v,\mathcal{X}_a\}$, where $\mathcal{X}_u \in \mathbb{R}^{T_u \times d_u}$}

\subsection{Graph-Structured Multimodal Fusion}
\label{sec:GSIFN}

We regard the low level temporal feature sequences $\{X_t,X_v,X_a\}$ as graph vertex sequences $\{\mathcal{V}_t,\mathcal{V}_v,\mathcal{V}_a\}$, where each time step is treated as a graph vertex. Then, concatenate vertices into a single sequence $\mathcal{V}_m$ = $[\mathcal{V}_t;\mathcal{V}_v;\mathcal{V}_a]^{\top}$. $\mathcal{V}_m$ is treated as the multimodal graph embedding. The architecture of Graph-Structured and Interlaced-Masked Multimodal Transformer Architecture (GsiT) is shown in Figure \ref{fig:gsit_im}.

\textbf{Graph Structure Construction} To start with, we utilize self attention mechanism as the basic theory to construct a naive fully connected graph. The attention weight matrix is regarded as the adjacency matrix with dynamic weights, which is constructed as follows:

\begin{equation}
\begin{aligned}
\label{eq:adjmat_all}
  \mathcal{A} &= (\mathcal{W}_q \mathcal{V}_m) \cdot (\mathcal{W}_k \mathcal{V}_m)^{\top} \\
        &= \begin{pmatrix}
          \mathcal{E}^{t,t} & \mathcal{E}^{t,v} & \mathcal{E}^{t,a} \\
          \mathcal{E}^{v,t} & \mathcal{E}^{v,v} & \mathcal{E}^{v,a} \\
          \mathcal{E}^{a,t} & \mathcal{E}^{a,v} & \mathcal{E}^{a,a} \\
        \end{pmatrix}
\end{aligned}
\end{equation} Where $\mathcal{E}^{i,j} \in \mathbb{R}^{T_i \times T_j}$, $\{i,j\}\in\{t,v,a\}$ is the adjacency matrix of the subgraph constructed by $\mathcal{V}_i$ and $\mathcal{V}_j$. 

The derivation process of detailed graph structure construction (from vertex to subgraph) is in Appendix \ref{derivation:graph_agg}.

\textbf{Interlaced Mask Mechanism} Interlaced Mask (IM) is a modal-wise mask mechanism, thus all of the elements in the mask matrix are subgraph adjacency matrices. The mask matrix is actually represented as a block matrix. Then the construction procedure of IM is described in detail. The computation procedure with IM is shown in Figure \ref{fig:gsit_im}.

To start with, in order to avoid the influence of intra-modal subgraph $\mathcal{E}^{i,j}_{\{i=j\}\in\{t,v,a\}}$, we apply modal-wise intra mask as shown in Equation \ref{naive_inter-mask}.

\begin{equation}
  \label{naive_inter-mask}
  \mathcal{M}_{inter} = \begin{pmatrix}
    \mathcal{J}^{t,t} & \mathcal{O}^{t,v} & \mathcal{O}^{t,a} \\
    \mathcal{O}^{v,t} & \mathcal{J}^{v,v} & \mathcal{O}^{v,a} \\
    \mathcal{O}^{a,t} & \mathcal{O}^{a,v} & \mathcal{J}^{a,a} \\
  \end{pmatrix}
\end{equation} Where $\mathcal{O}^{i,j} \in \mathbb{R}^{T_i \times T_j}$, $\mathcal{J}^{i,j} \in \mathbb{R}^{T_i \times T_j}$ denotes height of $T_i$ width of $T_j$ all respectively 0, negative infinity matrix.

$\mathcal{M}_{inter}$ can realize that cross-modal fusion is not affected by intra-modal subgraphs. However, in the fusion procedure, different modal sequences are not supposed to be recognized as the same sequence. It has to be noted that in the modal-wise mask matrix, only one subgraph matrix can be left unmasked on each row, otherwise temporal information disorder occurs. For instance, $\mathcal{E}^{t,v}$ and $\mathcal{E}^{t,a}$ are not masked yet, which makes their vertex sequence recognized as a whole, resulting in temporal information disorder. Therefore, we extend $\mathcal{M}_{inter}$ to the following two mask matrices, which is called as Interlaced-Inter-Fusion Mask (IFM).

\vspace{-0.3cm} 
\begin{equation}
  \begin{cases}
    \begin{aligned}
    &\mathcal{M}_{inter}^{forward} = \begin{pmatrix}
        \mathcal{J}^{t,t} & \mathcal{O}^{t,v} & \mathcal{J}^{t,a} \\
        \mathcal{J}^{v,t} & \mathcal{J}^{v,v} & \mathcal{O}^{v,a} \\
        \mathcal{O}^{a,t} & \mathcal{J}^{a,v} & \mathcal{J}^{a,a} \\
      \end{pmatrix} \\
    &\mathcal{M}_{inter}^{backward} = \begin{pmatrix}
        \mathcal{J}^{t,t} & \mathcal{J}^{t,v} & \mathcal{O}^{t,a} \\
        \mathcal{O}^{v,t} & \mathcal{J}^{v,v} & \mathcal{J}^{v,a} \\
        \mathcal{J}^{a,t} & \mathcal{O}^{a,v} & \mathcal{J}^{a,a} \\
      \end{pmatrix}
    \end{aligned}
  \end{cases}
\end{equation}

Based on the two matrices, two opposite uni-directional ring graphs can be constructed to achieve a complete fusion procedure.

\begin{equation}
  \begin{cases}
    \begin{aligned}
      &\mathcal{G}_{inter}^{forward} = \mathcal{S} \circ 
      \mathcal{D}(\mathcal{A} + \mathcal{M}_{inter}^{forward}) \\ 
      &\mathcal{G}_{inter}^{backward} = \mathcal{S} \circ 
      \mathcal{D}(\mathcal{A} + \mathcal{M}_{inter}^{backward}) 
    \end{aligned}
  \end{cases}
\end{equation} Where $\mathcal{S}$ denotes $SoftMax$ operation, $\mathcal{D}$ denotes $DropOut$ operation.

By now, $\mathcal{G}_{inter}^{forward}$ and $\mathcal{G}_{inter}^{backward}$ truly make graph embedding $\mathcal{V}_m$ graph structured. Both of the two matrices manage to aggregate the information of the three modalities without temporal disorder and intra-modal information influence. After aggregation, the fusion process is performed.

\begin{equation}
  \begin{cases}
    \begin{aligned}
      &\overline{\mathcal{V}}_m^{forward} = \mathcal{G}_{inter}^{forward} \cdot \mathcal{V}_m \\
      &\overline{\mathcal{V}}_m^{backward} = \mathcal{G}_{inter}^{backward} \cdot \mathcal{V}_m
    \end{aligned}  
  \end{cases}
\end{equation}

As shown in Figure \ref{fig:gsit_im}, two graph embeddings are constructed by IFM in two separated Transformers. Because the ring graph embedding contains all of the three modal information in the form of a unidirectional fusion cycle, it manages to share multimodal information through weight sharing strategy, thus a single Transformer can efficiently fuses three modal information. With two opposite unidirectional ring graph, on the perspective of fusion modal combinations, a complete fusion process is achieved.

After the modal fusion, the subgraph of the modal needs to be enhanced accordingly. At this time, the Intelaced-Intra-Enhancement Mask (IEM) can be constructed to realize the operation.

\begin{equation}
  \mathcal{M}_{intra} = \mathcal{J} - \mathcal{M}_{inter} 
\end{equation} Where $\mathcal{J}$ denotes a negative infinity matrix at the same size of $\mathcal{M}_{inter}$.

$\mathcal{M}_{intra}$ leave only intra-modal subgraphs visible, in order to enhance the fused embeddings.

After IEM construction, concatenate bidirectional features on the feature dimension, in order to compose the feature of two opposite unidirectional ring graph embedding into one bidirectional graph embedding.

\begin{equation}
\overline{\mathcal{V}}_m^{bidirection}=\parallel\overline{\mathcal{V}}_m^{\{forward,backward\}}
\end{equation} Where $\parallel$ denotes the concatenation operation on the feature dimension.

Utilizing the bidirectional graph embedding $\overline{\mathcal{V}}_m^{bidirection}$, the intra-modal enhancement graph could be constructed as below.

\begin{align}
  &\mathcal{A}_{fusion} = (\mathcal{W}_q^b * \overline{\mathcal{V}}_m^b)(\mathcal{W}_k^b * \overline{\mathcal{V}}_m^b)^\top \\
  &\mathcal{G}_{intra} = \mathcal{D} \circ \mathcal{S} (\mathcal{A}_{fusion} + \mathcal{M}_{intra}) 
\end{align} Where $\overline{\mathcal{V}}_m^{b}$ = $\overline{\mathcal{V}}_m^{bidirection}$, $\mathcal{W}_q^b$, $\mathcal{W}_k^b$ denotes the query, key projection weight of $\mathcal{V}_m^{b}$.

Then, we construct the final feature sequence.

\begin{equation}
  \overline{\mathcal{V}}_m = \mathcal{G}_{intra} \mathcal{W}_v^b \overline{\mathcal{V}}_m^{bidirection}
\end{equation} Where $\mathcal{W}_v^b$ denotes the value projection weight of $\mathcal{V}_m^{b}$.

Finally, the sequence is decomposed according to the length of the original feature sequence, then the final hidden states of different modals are taken to be concatenated on the feature dimension, in order to predict the final multimodal fusion representation.

% \begin{align}
%   &\mathcal{X}_m = \parallel\overline{\mathcal{V}}_m^{final} = \parallel[\overline{\mathcal{V}}_t; \overline{\mathcal{V}}_v; \overline{\mathcal{V}}_a]\\
%   \label{eq:30}
%   &\hat{y}_m = Predictor(\mathcal{X}_m) 
% \end{align} Where $\parallel$ denotes the concatenation operation on the feature dimension.

In the procedure of graph structured multimodal fusion, graph embedding and IM make sure every transformer layer fully fuses three modal information into the weight of it, thus significantly increase the efficiency of multimodal fusion. Since the layer weights are highly shared, the computational overhead of GsiT is much lower than that of previous methods.

The detailed generation algorithm of cross mask for inter-fusion and intra-enhancement is described in Appendix \ref{apdx:crossmaskgen}

\subsection{Self-Supervised Learning Framework}
\label{sec:self-supervision}

We integrate the unimodal label generation module (ULGM) into our method to capture modality-specific information. As shown in Figure \ref{fig:overview}, $\mathcal{X}_{\{t,v,a\}}$ are utilized to generate the unimodal labels $\hat{y}_{\{t,v,a\}}$, while the final hidden states $h_{\{t,v,a\}}$ generated during the prediction procedure along with the ground truth multimodal label are obtained by ULGM to define the positive and negetive centers with predicted unimodal labels and multimodal fusion representations. Afterwards, we calculate the relative distance from the representation of each modality to the positive and negative centers, and obtain the offset value from the unimodal label to the ground truth multimodal label to generate new unimodal label $y_{\{t,v,a\}}^i$ for $i^{th}$ epoch. In this way, it is more conducive to sentiment analysis to obtain differentiated information of different modalities while retaining the consistency of each modality.

Using the predicted results $\hat{y}_{\{m,t,v,a\}}$ and the ground truth multimodal label $y_m$ along with the generated labels $y_{\{t,v,a\}}$, we implemented a weighted loss to optimize our model.

The weighted loss is defined by Equation \ref{w_loss} whereas the unimodal loss for each modality is defined as Equation \ref{u_loss}

\begin{align}
  \label{w_loss}
  &\mathcal{L}_w = \sum_{u\in{\{m,t,v,a\}}}\mathcal{L}_u\\
  &\begin{aligned}
  \label{u_loss}
    &\mathcal{L}_u = \frac{\sum_{i=0}^{\mathcal{B}}{w_u^i * |\hat{y}_u^i - y_u^i|}}{\mathcal{B}} \\
    &w_u^i = \begin{cases}
      {1} & u = m \\
      \tanh{(|\hat{y}_u^i - \hat{y}_m^i|)} & u \in {\{t,v,a\}}
    \end{cases}  
  \end{aligned}
\end{align} Where $\mathcal{B}$ denotes the appointed batch size.

\begin{table*}[!htp]
  \caption{Comparison on CMU-MOSI and CMU-MOSEI.}
  \label{tab: Results_Compare_CMU-MOSI_CMU-MOSEI}
  \small
  \resizebox{1.0\linewidth}{!}
  {
    \begin{tabular}{lccccccccccccc}
      \toprule
      \multirow{2}{*}{Model} & \multicolumn{5}{c}{CMU-MOSI} & \multicolumn{5}{c}{CMU-MOSEI} & \multirow{2}{*}{Data State}\\
      & Acc-2\uparrow & F1\uparrow & Acc-7\uparrow & MAE\downarrow & Corr\uparrow & Acc-2\uparrow & F1\uparrow & Acc-7\uparrow & MAE\downarrow & Corr\uparrow  \\
      
      \midrule
      
      $\text{MulT}^*$ & 83.0 / - & 82.8 / - & 40.0 & 0.871 & 0.698 & 81.6 / - & 81.6 / - & 50.7 & 0.591 & 0.694 & Unaligned \\

      $\text{MTAG}^*$ & 82.3 / - & 82.1 / - & 38.9 & 0.866 & 0.722 & - / - &- / - &- &- &- & Unaligned \\

      $\text{MISA}^*$ & 81.8 / 83.4 &	81.7 / 83.6 &	42.3 &	0.783 & 0.761 & 83.6 / 85.5 & 83.8 / 85.3 & 52.2 & 0.555 & 0.756 & Unaligned \\

      $\text{HyCon-BERT}^*$ & - / 85.2 & - / 85.1 & 46.6 & 0.713 & 0.790 & - / 85.4 & - / 85.6 & 52.8 & 0.601 & 0.776 & Aligned \\

      $\text{TETFN}^*$ & 84.1 / 86.1 & 83.8 / 86.1 & - & 0.717 & 0.800	& 84.3 / 85.2 & 84.2 / 85.3 & - & 0.551 & 0.748 & Unaligned \\
      
      $\text{ConFEDE}^*$ & \uuline{84.2} / \uuline{85.5} & \uuline{84.1} / \uuline{85.5} & 42.3 & 0.742 & 0.784 & 81.7 / \uuline{85.8} & 82.2 / \uuline{85.8} & \textbf{54.9} & \textbf{0.522} & \textbf{0.780} & Unaligned \\

      $\text{MMIN}^*$ & 83.5 / 85.5 & 83.5 / 85.5 & - & 0.741 & 0.795 & 83.8 / 85.9 & 83.9 / 85.8 & - & 0.542 & 0.761 & Unaligned \\

      $\text{MTMD}^*$ & 84.0 / \textbf{86.0} & 83.9 / \textbf{86.0} & \uuline{47.5} & \textbf{0.705} & \uuline{0.799} & \uuline{84.8} / \uuline{86.1} & \uuline{84.9} / \uuline{85.9} & \uuline{53.7} & \uuline{0.531} & \uuline{0.767} & Unaligned \\

      \midrule
      
      $\text{MulT}^\dagger$ & 79.6 / 81.4 & 79.1 / 81.0 & 36.2 & 0.923 & 0.686 & 78.1 / 83.7 & 78.9 / 83.7 & 53.4 & 0.559 & 0.740 & 4.362 & 105.174 \\ 
      
      $\text{MAG-BERT}^\dagger$ & 82.2 / 84.3 & 82.1 / 84.2 & 46.4 & 0.722 & 0.785 & 77.7 / 84.0 & 78.6 / 84.1 &	\uuline{53.9} & \uuline{0.536} & 0.755 & Aligned \\

      $\text{CENet-BERT}^\dagger$ &	82.8 / 84.5 & 82.7 / 84.5 &	45.2 & 0.736 & 0.793 & 81.7 / 82.3 &	81.6 / 81.9 & 52.0 & 0.576 & 0.711 & Aligned \\

      $\text{Self-MM}^\dagger$ & 82.2 / 83.5 & 82.3 / 83.6 & 43.9 & 0.758 & 0.792 & 80.8 / 85.0 & 81.3 / 84.9 & \uuline{53.3} & \uuline{0.539} & \uuline{0.761} & Unaligned \\
      
      $\text{TETFN}^\dagger$ & 82.4 / 84.0	& 82.4 / 84.1	& 46.1	& 0.749 & 0.784 & 81.9 / 84.3 & 82.1 / 84.1 & 52.7 & 0.576 & 0.728 & Unaligned \\ 

      \midrule

      \textbf{GSIFN} 
      & \textbf{85.0} / \textbf{86.0} & \textbf{85.0} / \textbf{86.0} & \textbf{48.3} & \uuline{0.707} & \textbf{0.801} & \textbf{85.0} / \textbf{86.3} & \textbf{85.1} / \textbf{86.2} & \uuline{53.4} & \uuline{0.538} & \uuline{0.767} & Unaligned \\
      \bottomrule
    \end{tabular}
  }

\end{table*}

\begin{table}[!htp]
  \caption{Comparison on CH-SIMS.}
  \label{tab: Results_Compare_SIMS}
  \resizebox{0.48\textwidth}{!}{
    \begin{tabular}{lcccccc}
      \toprule
      \multirow{2}{*}{Model} & \multicolumn{6}{c}{CH-SIMS} \\
      &Acc-2\uparrow & Acc-3\uparrow & Acc-5\uparrow & F1\uparrow & MAE\downarrow & Corr\uparrow\\
      \midrule
      TFN   &77.7  &66.3  &42.7  &77.7  &0.436 &0.582\\
      MFN   &77.8  &65.4  &38.8  &77.6  &0.443 &0.566\\
      MulT	&77.8	&65.3	&38.2	&77.7	&0.443	&0.578\\
      MISA	&75.3	&62.4	&35.5	&75.4	&0.457	&0.553\\
      Self-MM	&78.1	&65.2	&41.3	&78.2	&0.423	&0.585\\
      % \midrule
      TETFN	&78.0	&64.4	&42.9	&78.0	&0.425	&0.582\\
      \midrule
      \textbf{GSIFN} & \textbf{80.5} & \textbf{67.2} & \textbf{45.5} & \textbf{80.7} & \textbf{0.397} & \textbf{0.619} \\
      \bottomrule
    \end{tabular}
    }
  
\end{table}

\section{Experiment}

\subsection{Datasets}

We evaluate our model on three benchmarks, CMU-MOSI \cite{arxiv:mosi}, CMU-MOSEI \cite{acl:mosei} and CH-SIMS \cite{acl:ch-sims}. These datasets provide aligned (CMU-MOSI, CMU-MOSEI) and unaligned (all) mutlimodal data (text, vision and audio) for each utterrance. Here, we give a brief introduction to the above datasets.
Further details on the datasets are described in Appendix \ref{apdx:dataset}

\subsection{Evaluation Criteria}

Following prior works, several evaluation metrics are adopted. Binary classification accuracy (Acc-2), F1 Score (F1), three classification accuracy (Acc-3), five classification accuracy (Acc-5), seven classification accuracy (Acc-7), mean absolute error (MAE), and the correlation of the model's prediction with human (Corr). Specially, Acc-3 and Acc-5 are applied only for CH-SIMS dataset, Acc-2 and F1 are calculated in two ways: negative/non-negative(NN) and negative/positive(NP) on MOSI and MOSEI datasets, respectively.

\subsection{Baselines}

For CMU-MOSI and CMU-MOSEI, we choose MAG-BERT, MulT, MTAG, MISA, Self-MM, CENet, TETFN, ConFEDE and MTMD as baselines. As for CH-SIMS, on account of the data state of it is all unaligned, the baselines are different from those of the former two datasets, TFN, MFN, MISA, MulT, Self-MM and TETFN are chosen.
For a more detailed introduction of the baseline models, please refer Appendix \ref{apdx:baseline}

\subsection{Results}

The performance comparison of all methods on MOSI, MOSEI and CH-SIMS are summarized in Table \ref{tab: Results_Compare_CMU-MOSI_CMU-MOSEI} and Table \ref{tab: Results_Compare_SIMS}. 

In Table \ref{tab: Results_Compare_CMU-MOSI_CMU-MOSEI}, for a fair comparison in CMU-MOSI and CMU-MOSEI, we split models into two categories for data state: Unaligned and Aligned, $^\dagger$ denotes that the model is sourced from the GitHub page\textsuperscript{\ref{mmsa}} and the scores are reproduced, $^*$ denotes the result is obtained directly from the original paper. For Acc-2 and F1, the left of the "/" corresponds to "negative/non-negetive" and the right corresponds to "negative/positive". For all metrics, the best results are highlighted in bold, and the weaker but still excellent results are double-underlined. 

In Table \ref{tab: Results_Compare_SIMS}, the best results are highlighted in bold, all of the models are sourced from the GitHub page\textsuperscript{\ref{mmsa}} and the scores are reproduced.


\textbf{Analysis on CMU-MOSI} As shown in the Table \ref{tab: Results_Compare_CMU-MOSI_CMU-MOSEI}, the proposed GSIFN surpasses baselines on almost all the metrics on CMU-MOSI dataset. On Acc-2(NN\&NP), F1(NN\&NP), Acc-7 and Corr, it outperforms all the baselines, especially on Acc-2(NN), F1(NN) and Acc-7, GSIFN achieves a relative improvement of 0.8\%, 0.9\%, and 0.8\% than the best performance of baselines. As for the MAE and Corr, it performs similar with the best baseline MTMD, with a 0.002 reduction on MAE and a 0.002 improvement on Corr. 

\textbf{Analysis on CMU-MOSEI} As shown in the Table \ref{tab: Results_Compare_CMU-MOSI_CMU-MOSEI}, GSIFN achieves the optimal performance on Acc-2(NN\&NP) and F1(NN\&NP) where performs admirably on Acc-2(NN) and F1(NN), which surpasses not only all the baselines an average of 3.3\% on ACC-2(NN) and 3.0\% on F1(NN) but also the best baseline MTMD 0.2\% on both Acc-2(NN) and F1(NN). The results of Acc-7, MAE and Corr manage to reach a excellent level among all the baselines, although they were slightly weaker than the best baseline ConFEDE. 

\footnotetext[1]{\url{https://github.com/thuiar/MMSA} \label{mmsa}}

\textbf{Analysis on CH-SIMS} As shown in the Table \ref{tab: Results_Compare_SIMS}, GSIFN achieves optimal results over all baselines, with at least 2.4\% in Acc-2, 2.0\% in Acc-3, 1.6\% in Acc-5, 2.5\% in F1, 0.026 in MAE, 0.034 in Corr, all of which are tremendous improvement.

\subsection{Ablation Study}

In this session, we will discuss our ablation study and its results in detail, which are divided into four parts in Table \ref{tab: Ablation Study}: Module Ablation, Modality Ablation and Pretrained Language Model Ablation.

\textbf{Module Ablation} There are three main modules in our model, including Graph-Structured Interlaced-Masked Multimodal Transformer (GsiT) for multimodal fusion, extended LSTM with matrix memory (mLSTM) for vision, audio temporal enhancement, Unimodal Label Generation Module (ULGM) for self supervision. In Table \ref{tab: Ablation Study} part module Ablation, w/o denotes the absence of corresponding module in the model. 

The results in Table \ref{tab: Ablation Study} indicates module GsiT and ULGM are necessary for achieving state-of-the-art performance. GsiT module constructs the graph structure of three modalities , without module GsiT, the performance of the whole model has a substantial decrease in all metrics, especially 1.2\% on Acc-2(NN), 1.8\% on F1(NN), 1.8\% on Acc-7, and 0.035 on MAE. Without module mLSTM, the performance weakens mainly on fine-grained metrics, 1.1\% on Acc-7 and 0.023 on MAE. Without module ULGM, the performance weakens on all the metrics, especially on binary and seven classification task, 1.6\%/1.2\% on Acc-2 and 1.6\%/1.2\% on F1, 1.6\% on Acc-7.

\begin{table}[!htp]
  \caption{Ablation study on CMU-MOSI. Note: F denotes finetuning pretrained language models, NF denotes not finetuning }
  \label{tab: Ablation Study}
  \resizebox{0.48\textwidth}{!}  
  {
    \begin{tabular}{lcccccc}
      \toprule
      \multirow{2}{*}{Description} & \multicolumn{5}{c}{CMU-MOSI} \\
       & Acc-2\uparrow & F1\uparrow & Acc-7\uparrow & MAE\downarrow & Corr\uparrow \\
      \midrule
      \multicolumn{6}{c}{Module Ablation} \\
      \midrule
      \text{GSIFN} & \textbf{85.0} / \textbf{86.0} & \textbf{85.0} / \textbf{86.0} & \textbf{48.3} & \textbf{0.707} & \textbf{0.801} \\
      \text{w/o GsiT} & 83.8 / 85.5 & 83.2 / 85.7 & 46.5 & 0.742 & 0.790 \\
      \text{w/o mLSTM} & 84.6 / 86.0 & 84.5 / 86.0 & 47.2 & 0.730 & 0.792 \\
      \text{w/o ULGM} & 83.4 / 84.8 & 83.4 / 84.8 & 46.7 & 0.711 & \textbf{0.801} \\
      \midrule
      \multicolumn{6}{c}{Pretrained Language Model Ablation} \\
      \midrule
      \text{BERT(F)} & \textbf{85.0} / \textbf{86.0} & \textbf{85.0} / \textbf{86.0} & \textbf{48.3} & \textbf{0.707} & \textbf{0.801} \\
      \text{BERT(NF)} & 83.8 / 85.7 & 83.7 / 85.6 & 46.1 & 0.731 & 0.796 \\
      \bottomrule
    \end{tabular}
  }
  % \vspace{-1.0cm}
\end{table}

\textbf{Pretrained Language Model Ablation} The experiment on whether or not finetuning BERT is shown in Table \ref{tab: Ablation Study}, part Pretrained Language Model Ablation. The result shows that BERT finetuning is quite useful to GSIFN.

\begin{figure*}[htpb]
\begin{center}
  \includegraphics[width=\linewidth]{fig/Map_1.png}
\end{center}
  \caption{Example of Alignment.}
\label{fig:alignment}
\end{figure*}

\begin{table*}[!htp]
  \caption{Comparison of GsiT and MulT on CMU-MOSI and CMU-MOSEI.}
  \label{tab: GsiT&MulT_Compare_CMU-MOSI_CMU-MOSEI}
  \small
  \resizebox{1.0\linewidth}{!}
  {
    \begin{tabular}{lcccccccccccccc}
      \toprule
      \multirow{2}{*}{Model} & \multicolumn{5}{c}{CMU-MOSI} & \multicolumn{5}{c}{CMU-MOSEI} & \multirow{2}{*}{Parameters(M)} & \multirow{2}{*}{FLOPS(G)}\\ 
      & Acc-2\uparrow & F1\uparrow & Acc-7\uparrow & MAE\downarrow & Corr\uparrow & Acc-2\uparrow & F1\uparrow & Acc-7\uparrow & MAE\downarrow & Corr\uparrow  \\
      \midrule
      MulT & 79.6 / 81.4 & 79.1 / 81.0 & 36.2 & 0.923 & 0.686
      & 78.1 / 83.7 & 78.9 / 83.7 & 53.4 & 0.559 & 0.740 & 4.362 & 105.174 \\ 
      GsiT & 83.4 / 84.9 & 83.4 / 85.0 & 45.5 & 0.716 & 0.803
      & 84.1 / 86.3 & 84.4 / 86.3 & 53.5 & 0.539 & 0.774 & 0.891 & 25.983 \\
      \bottomrule
    \end{tabular}
  }
\end{table*}

\begin{table}[!htp]
  \caption{Graph Structure Case Study on CMU-MOSI}
  \label{tab: Graph Structure}
  \resizebox{0.48\textwidth}{!}  
  {
    \begin{tabular}{lcccccc}
      \toprule
      \multirow{2}{*}{Description} & \multicolumn{5}{c}{CMU-MOSI} \\
       & Acc-2\uparrow & F1\uparrow & Acc-7\uparrow & MAE\downarrow & Corr\uparrow \\
      \midrule
      \text{Orginal} & \textbf{85.0} / \textbf{86.0} & \textbf{85.0} / \textbf{86.0} & \textbf{48.3} & \textbf{0.707} & \textbf{0.801} \\
      \text{Structure-1} & 82.4 / 84.0 & 82.3 / 84.0 & 46.5 & 0.712 & 0.792 \\
      \text{Structure-2} & 83.8 / 85.7 & 83.7 / 85.6 & 46.1 & 0.731 & 0.796 \\
      \text{Structure-3} & 83.4 / 85.1 & 83.3 / 85.1 & 45.5 & 0.727 & 0.793 \\
      \text{Self-Only} & 81.6 / 83.2 & 81.7 / 83.3 & 43.3 & 0.750 & 0.791 \\
      \bottomrule
    \end{tabular}
  }
  % \vspace{-1.0cm}
\end{table}

\begin{table}[!htp]
  \caption{The Computational Overhead of Different Vision/Audio Modality Enhancement Models}
  \label{tab: Param}
  \resizebox{0.48\textwidth}{!}  
  {
    \begin{tabular}{lccccc}
      \toprule
      Model & mLSTM(V) & mLSTM(A) & ViT & Wav2Vec & Whisper \\
      \midrule
      \text{Parameters(M)} & 0.439 & 0.439 & 127.272 & 94.395 & 17.120 \\
      \text{FLOPS(G)} & 1.674 & 1.252 & 35.469 & 68.543 & 315.128 \\
      \bottomrule
    \end{tabular}
  }
  % \vspace{-1.0cm}
\end{table}

\textbf{Modality Ablation} Modality ablation study is shown in Appendix \ref{apdx:modality_ablation}

\subsection{Further Analysis}

We discuss the graph structure construction, alignment efficiency of GsiT, GsiT performance comparison with MulT and mLSTM efficiency in this section.

\textbf{Graph Structure Selection} The structure of the graph has a significant impact on the performance of the model, so we conduct an ablation study on its graph structure. The graph structure of the three modalities can only be constructed in the following four structures: Original strucuture (Org): \{$t$ -> $v$ -> $a$\} \& \{$a$ -> $v$ -> $t$\}; Structure-1: \{$a$ -> $v$ -> $a$, $a$ -> $t$\} \& \{$v$ -> $t$ -> $v$, $t$ -> $a$\}; Structure-2: \{$v$ -> $t$ -> $v$, $v$ -> $a$\} \& \{$a$ -> $t$ -> $a$, $a$ -> $v$\}; Structure-3: \{$a$ -> $v$ -> $a$, $v$ -> $t$\} \& \{$a$ -> $t$ -> $a$, $t$ -> $v$\}. As a contrast, we constructed a graph with only intra-mask which is diordered in multimodal temporal information: Self-Only: only mask the intra-modal subgraphs. The results are in Table \ref{tab: Graph Structure}
The cross masks of all the structures is described in detail in Appendix \ref{apdx:gs}

\textbf{Alignment of Sequences} An example of alignment efficiency of GSIFN is shown in Figure \ref{fig:alignment}. We choose alignment vision to text and audio to text as an example, these two groups are produced from two different graph embeddings. As can be seen from Figure \ref{fig:alignment}, GSIFN effectively and comprehensively compose the semantic of three modal together.

\textbf{GsiT and MulT} MulT mainly uses cross-attention to realize efficient modal sequence alignment and fusion. Like the core module of GSIFN, which is GsiT, MulT realizes directional fusion and post-fusion enhancement between modes. However, GsiT abstracts the whole process into a complete graph structure, uses interlaced mask to realize the graph structure construction, and merges forward and reverse processes respectively. The fusion process of this process realizes weight sharing with Transformer, avoids high-level weight isolation, avoids overfitting, and achieves better weight regularization.

For a fair comparison, we reproduced the MulT and trained MulT and GsiT with the same hyper parameters. The experiments are shown in the Table \ref{tab: GsiT&MulT_Compare_CMU-MOSI_CMU-MOSEI}. GsiT outperforms MulT significantly in all metrics. What is more, the number of parameters and floating-point operations per second (FLOPS) of GsiT are much lower than MulT.

\textbf{Vision/Audio Encoder Efficiency} As shown in Table \ref{tab: GsiT&MulT_Compare_CMU-MOSI_CMU-MOSEI}, the number of parameters (Params) and floating-point operations per second (FLOPS) of widely used non-verbal modal feature extractors. Vision Transformer (ViT), Wav2Vec and Whisper are employed to extract high quality feature. We employ mLSTM to enhance low quality feature extracted by LibROSA, OpenFace, etc. The Params and FLOPS of mLSTMs is way lower than ViT, Wav2Vec and Whisper.



\section{Conclusion}

In this paper, we propose GSIFN, a Graph-Structured and Interlaced Multimodal Transformer Based Fusion Network. The core components of GSIFN are (i) Graph-structured and Interlaced Masked Multimodal Transformer, which uses the interlaced mask mechanism to build multi-modal graph embedding and implement Transformer's modal wise graph structure (ii) Self-supervised learning framework, supplemented by non-text modal feature enhancement module using mLSTM. The experimental results show that GSIFN reduces the computational overhead and is better than the previous SOTA algorithm.

\section*{Limitations}

Our GSIFN lacks pre-training for different modal data in the feature extraction part, and does not deal with the overpopulated part of the data, resulting in too many redundant and uninformative vertices in the graph structure. This has an impact on our performance on fine-grained tasks in the dataset, such as MAE and Acc-7, which is not outstanding compared to past methods.

% Entries for the entire Anthology, followed by custom entries
\normalem
\bibliography{custom}
% \bibliographystyle{coling_natbib}


\clearpage
\appendix

\section{Experiment}

\subsection{Modality Ablation}
\label{apdx:modality_ablation}
\begin{table}[!htp]
  \caption{Modality Ablation Study on CMU-MOSI. }
  \label{tab: Modality Ablation}
  \resizebox{0.48\textwidth}{!}  
  {
    \begin{tabular}{lcccccc}
      \toprule
      \multirow{2}{*}{Description} & \multicolumn{5}{c}{CMU-MOSI} \\
       & Acc-2\uparrow & F1\uparrow & Acc-7\uparrow & MAE\downarrow & Corr\uparrow \\
      \midrule
      \multicolumn{6}{c}{Fusion Modality Ablation} \\
      \midrule
      \text{M(T,V,A)} & \textbf{85.0} / \textbf{86.0} & \textbf{85.0} / \textbf{86.0} & \textbf{48.3} & \textbf{0.707} & \textbf{0.801} \\
      \text{M(T,V)}  & 84.3 / 85.5 & 84.2 / 85.5 & 45.5 & 0.720 & 0.797 \\
      \text{M(T,A)}  & 84.3 / 85.7 & 84.3 / 85.7 & 47.2 & 0.704 & 0.800 \\
      \text{M(V,A)}  & 59.8 / 60.2 & 59.7 / 60.3 & 17.9 & 1.344 & 0.196 \\
      \text{M(T)}   & 83.1 / 84.8 & 83.0 / 84.7 & 47.5 & 0.715 & 0.786 \\
      \text{M(V)}   & 59.2 / 59.8 & 58.9 / 59.6 & 16.8 & 1.372 & 0.141 \\
      \text{M(A)}   & 60.4 / 61.3 & 59.0 / 60.0 & 21.3 & 1.322 & 0.236 \\
      \midrule
      \multicolumn{6}{c}{ULGM Modality Ablation} \\
      \midrule
      \text{M+T+V+A} & \textbf{85.0} / \textbf{86.0} & \textbf{85.0} / \textbf{86.0} & \textbf{48.3} & \textbf{0.707} & \textbf{0.801} \\
      \text{M+T+V} & 84.4 / 85.7 & 84.3 / 85.7 & 44.5 & 0.742 & 0.742 \\
      \text{M+T+A} & 83.9 / 85.7 & 83.7 / 85.6 & 46.1 & 0.731 & 0.796 \\
      \text{M+V+A} & 83.8 / 85.2 & 83.8 / 85.3 & 44.6 & 0.748 & 0.794 \\
      \text{M+T}  & 83.4 / 85.7 & 83.3 / 85.6 & 45.0 & 0.731 & 0.796 \\
      \text{M+V}  & 83.5 / 85.4 & 83.5 / 85.4 & 45.8 & 0.724 & \textbf{0.801} \\
      \text{M+A}  & 82.5 / 84.6 & 82.4 / 84.6 & 46.1 & 0.709 & 0.800 \\
      \text{M}   & 83.4 / 84.8 & 83.4 / 84.8 & 46.7 & 0.711 & \textbf{0.801} \\
      \bottomrule
    \end{tabular}
  }
  % \vspace{-1.0cm}
\end{table}

\textbf{Fusion Modality Ablation} In GSIFN, the multimodal representation (M) is used for the final classification task task. In the original case, M is composed of unimodal text (T), vision (V), and audio (A). In order to fully investigate the influence of the combined form of multimodal representation on the representation ability of the whole model, we designed the Modality Ablation study, which contains the three-modal case: M(T,V,A); the two-modal case: M(T,V), M(T,A), M(V,A); and the single-modal case: M(T), M(V), M(A). Note that the structure of the model in the single-modal case is already missing, and the graph structured attention degenerates to naive multi-head self-attention.

\textbf{ULGM Modality Ablation} In our proposed Self-Supervised Learning Framework, multimodality (M) is used for classification, and unimodal text (T), vision (V), and audio (A) are used to generate unimodal labels in ULGM to ensure that the model learns a robust representation of the multimodal data. To fully analyze the importance of each modality in the model, we design modality selection experiments for self-supervised modality adoption. There are three modal label generation: M+T+V+A; two modal label generation: M+T+V, M+T+A, M+V+A; single modal label generation: M+T, M+V, M+A; and no label generation: M.

\subsection{Graph Structures}
\label{apdx:gs}

The graph structures constructed in ablation study in the part Graph Structure Cases of Table \ref{tab: GsiT&MulT_Compare_CMU-MOSI_CMU-MOSEI}. Cross masks for each different graph structure are as follows:

\textbf{Original Structure (Bidirectional)}:

\vspace{-0.3cm} 
\begin{equation}
  \begin{cases}
    \begin{aligned}
    &\mathcal{M}_{inter}^{forward} = \begin{pmatrix}
        \mathcal{J}^{t,t} & \mathcal{O}^{t,v} & \mathcal{J}^{t,a} \\
        \mathcal{J}^{v,t} & \mathcal{J}^{v,v} & \mathcal{O}^{v,a} \\
        \mathcal{O}^{a,t} & \mathcal{J}^{a,v} & \mathcal{J}^{a,a} \\
      \end{pmatrix} \\
    &\mathcal{M}_{inter}^{backward} = \begin{pmatrix}
        \mathcal{J}^{t,t} & \mathcal{J}^{v,t} & \mathcal{O}^{a,t} \\
        \mathcal{O}^{v,t} & \mathcal{J}^{v,v} & \mathcal{J}^{v,a} \\
        \mathcal{J}^{a,t} & \mathcal{O}^{a,v} & \mathcal{J}^{a,a} \\
      \end{pmatrix}
    \end{aligned}
  \end{cases}
\end{equation}

\textbf{Structure-1}:

\vspace{-0.3cm} 
\begin{equation}
  \begin{cases}
    \begin{aligned}
    &\mathcal{M}_{inter}^{forward} = \begin{pmatrix}
        \mathcal{J}^{t,t} & \mathcal{J}^{t,v} & \mathcal{O}^{t,a} \\
        \mathcal{J}^{v,t} & \mathcal{J}^{v,v} & \mathcal{O}^{v,a} \\
        \mathcal{O}^{a,t} & \mathcal{J}^{a,v} & \mathcal{J}^{a,a} \\
      \end{pmatrix} \\
    &\mathcal{M}_{inter}^{backward} = \begin{pmatrix}
        \mathcal{J}^{t,t} & \mathcal{O}^{v,t} & \mathcal{J}^{t,a} \\
        \mathcal{O}^{v,t} & \mathcal{J}^{v,v} & \mathcal{J}^{v,a} \\
        \mathcal{O}^{a,t} & \mathcal{J}^{a,v} & \mathcal{J}^{a,a} \\
      \end{pmatrix}
    \end{aligned}
  \end{cases}
\end{equation}

\textbf{Structure-2}:

\vspace{-0.3cm} 
\begin{equation}
  \begin{cases}
    \begin{aligned}
    &\mathcal{M}_{inter}^{forward} = \begin{pmatrix}
        \mathcal{J}^{t,t} & \mathcal{O}^{t,v} & \mathcal{J}^{t,a} \\
        \mathcal{O}^{v,t} & \mathcal{J}^{v,v} & \mathcal{J}^{v,a} \\
        \mathcal{J}^{a,t} & \mathcal{O}^{a,v} & \mathcal{J}^{a,a} \\
      \end{pmatrix} \\
    &\mathcal{M}_{inter}^{backward} = \begin{pmatrix}
        \mathcal{J}^{t,t} & \mathcal{J}^{v,t} & \mathcal{O}^{t,a} \\
        \mathcal{J}^{v,t} & \mathcal{J}^{v,v} & \mathcal{O}^{v,a} \\
        \mathcal{O}^{a,t} & \mathcal{J}^{a,v} & \mathcal{J}^{a,a} \\
      \end{pmatrix}
    \end{aligned}
  \end{cases}
\end{equation}

\textbf{Structure-3}:

\vspace{-0.3cm} 
\begin{equation}
  \begin{cases}
    \begin{aligned}
    &\mathcal{M}_{inter}^{forward} = \begin{pmatrix}
        \mathcal{J}^{t,t} & \mathcal{O}^{t,v} & \mathcal{J}^{t,a} \\
        \mathcal{J}^{v,t} & \mathcal{J}^{v,v} & \mathcal{O}^{v,a} \\
        \mathcal{J}^{a,t} & \mathcal{O}^{a,v} & \mathcal{J}^{a,a} \\
      \end{pmatrix} \\
    &\mathcal{M}_{inter}^{backward} = \begin{pmatrix}
        \mathcal{J}^{t,t} & \mathcal{J}^{v,t} & \mathcal{O}^{t,a} \\
        \mathcal{O}^{v,t} & \mathcal{J}^{v,v} & \mathcal{J}^{v,a} \\
        \mathcal{O}^{a,t} & \mathcal{J}^{a,v} & \mathcal{J}^{a,a} \\
      \end{pmatrix}
    \end{aligned}
  \end{cases}
\end{equation}

\textbf{Self-Only}: 

\vspace{-0.3cm} 
\begin{equation}
  \mathcal{M}_{inter} = \begin{pmatrix}
    \mathcal{J}^{t,t} & \mathcal{O}^{t,v} & \mathcal{O}^{t,a} \\
    \mathcal{O}^{v,t} & \mathcal{J}^{v,v} & \mathcal{O}^{v,a} \\
    \mathcal{O}^{a,t} & \mathcal{O}^{a,v} & \mathcal{J}^{a,a} \\
  \end{pmatrix}
\end{equation}

\subsection{Alignment}

Specifically, a real-time example of original structure is shown in Figure \ref{fig:map_complete}. 

\begin{figure}[htpb]
\begin{center}
   \includegraphics[width=\columnwidth]{fig/map complete.png}
   \caption{Attention map split by interlaced masks.}
   \label{fig:map_complete}
% \vspace{-1.2cm}
\end{center}
\end{figure}

\section{Datasets}
\label{apdx:dataset}

Brief introduction to the three chosen datasets are as follows.

\textbf{CMU-MOSI}: The CMU-MOSI is a commonly used dataset for human multimodal sentiment analysis. It consists of 2,198 short monologue video clips (each clip lasts for the duration of one sentence) expressing the opinion of the speaker inside the video on a topic such as movies. The utterances are manually annotated with a continuous opinion score between [−3, +3], [−3: highly negative, −2 negative, −1 weakly negative, 0 neutral, +1 weakly positive, +2 positive, +3 highly positive].

\textbf{CMU-MOSEI}: The CMU-MOSEI is an improved version of CMU-MOSI. It contains 23,453 annotated video clips (about 10 times more than CMU-MOSI) from 5,000 videos, 1,000 different speakers, and 250 different topics. The number of discourses, samples, speakers, and topics is also larger than CMU-MOSI. The range of labels taken for each discourse is consistent with CMU-MOSI.

\textbf{CH-SIMS}: The CH-SIMS includes the same modalities in Mandarin: audio, text, and video, collected from 2281 annotated video segments. It includes data from TV shows and movies,making it culturally distinct and diverse, and provides multiple labels for the same utterance based on different modalities, which adds an extra layer of complexity and richness to the data.

\section{Baselines}
\label{apdx:baseline}

The introduction to baseline models is as follows.

\textbf{TFN}: The Tensor Fusion Network (TFN) uses modality embedding subnetwork and tensor fusion to learn intra- and inter-modality dynamics.

\textbf{MFN}: The Memory Fusion Network (MFN) explicitly accounts for both interactions in a neural architecture and continuously models them through time.

\textbf{MAG-BERT}: The Multimodal Adaptation Gate for Bert (MAG-Bert) incorporates aligned nonverbal information to text representation within Bert.

\textbf{MulT}: The Multimodal Transformer (MulT) uses cross-modal transformer based on cross-modal attention to make modality translation.

\textbf{MTAG}: The Modal-Temporal Attention Graph (MTAG) is a graph neural network model that incorporates modal attention mechanisms and dynamic pruning techniques to effectively capture complex interactions across modes and time, achieving a parametrically efficient and interpretable model.

\textbf{MISA}: The Modality-Invariant and -Specific Representations (MISA) projects representations into modality-sprcific and modality-invariant spaces and learns distributional similarity, orthogonal loss, reconstruction loss and task prediction loss

\textbf{Self-MM}: Learning Modality-Specific Representations with Self-Supervised Multi-Task Learning (Self-MM) [8] designs a multi- and a uni- task to learn inter-modal consistency and intra-modal specificity

\textbf{CENet}: Cross-Modal Enhancement Network (CENet) uses K-Means clustering to cluster the visual and audio modes into multiple tokens to realize the generation of the corresponding embedding, thus improving the representation ability of the two auxiliary modes and realizing a better BERT fine-tuning migration gate

\textbf{TETFN}: Text Enhanced Transformer Fusion Network (TETFN) strengthens the role of text modes in multimodal information fusion through text-oriented cross-modal mapping and single-modal label generation, and uses Vision-Transformer pre-training model to extract visual features

\textbf{ConFEDE}: Contrastive Feature Decomposition (ConFEDE) constructs a unified learning framework that jointly performs contrastive representation learning and contrastive feature decomposition to enhance representation of multimodal information.

\textbf{MTMD}: Multi-Task Momentum Distillation (MTMD) treats the modal learning process as multiple subtasks and knowledge distillation between teacher network and student network effectively reduces the gap between different modes, and uses momentum models to explore mode-specific knowledge and learn robust multimodal representations through adaptive momentum fusion factors.



\section{Aggregation of Modal Subgraphs}

\subsection{How to Aggregate Subgraphs?}

\label{derivation:graph_agg}

The derivation of graph aggregation from vertex to subgraph.

\textbf{Vertex Aggregation} Assuming a set of vertex features, $\mathcal{V}=\{v_1, v_2,\dots,v_N\}$, $v_i \in \mathbb{R}^D$, where $N$ is the number of vertices, and D is the feature dimension in each vertex. 

From previous works \cite{iclr:gat, iclr:gatv2}, the GAT is defined as follows. GAT performs self-attention on the vertices, which is a shared attentional mechanism $a: \mathbb{R}^{D^{'}} \times \mathbb{R}^{D} \rightarrow \mathbb{R}$ computes attention coefficients. Before that, a shared linear transformation, parameterized by a weight matrix, $\mathbf{W} \in \mathbb{R}^{D^{'} \times D} $. 

\begin{equation}
   e^{i,j} = a(\mathbf{W}v_i, \mathbf{W}v_j) = (\mathbf{W}v_i) \cdot (\mathbf{W}v_i)^\top
\end{equation}

$e^{ij}$ indicates the importance of vertex $j$'s feautures to vertex $i$. In the most general fomulation, the model allows vertex to attend on every other vertex, which drops all structural information. GAT inject the graph structure into the mechanism by performing masked attention: it only compute $e^{ij}$ for vertex $j \in \mathcal{N}_i$, where $\mathcal{N}_i$ is some neighbor of vertex i in the graph. To make coefficients easily comparable across different vertices, GAT normalize them across all choices of $j$ using the softmax function ($\mathcal{S}$):

\begin{equation}
  \alpha^{i,j} = \mathcal{S}_j(e^{i,j}) = \frac{\exp(e^{i,j})}{\sum_{k\in\mathcal{N}_i}\exp(e^{i,k})}
\end{equation}

Unlike GAT, whose attention machanism $a$ is a single-layer feedforward neural network, we directly employ multi-head self-attention mechanism as the aggregation algorithm.

Therefore, the final output features for every vertex is defined as follows.

\begin{equation}
  \overline{v}_i = \sum_{j\in\mathcal{N}_i}\alpha^{i,j}\mathbf{W}v_j
\end{equation} Where $\sigma$ indicates the sigmoid nonlinearity.

Then, we extend the mechanism to multi-head attention.

\begin{equation}
  \overline{v}_i = \parallel^K_{k=1} \sum_{j\in\mathcal{N}_i}\alpha^{i,j}_k\mathbf{W}_k v_j
\end{equation} Where $\parallel$ represents the concatenation operation.

\textbf{From Vertex to Subgraph} Assuming two sets of vertices $\mathcal{V}_i=\{v_1^i, v_2^i,\dots,v_{N_i}^i\}$, $v_m^i \in \mathbb{R}^{D_i}$ and $\mathcal{V}_j=\{v_1^j, v_2^j,\dots,v_{N_j}^j\}$, $v_n^j \in \mathbb{R}^{D_j}$. Where $N_{\{i,j\}}$ is the number of vertices of $\mathcal{V}_{\{i,j\}}$, $D_{\{i,j\}}$ is the feature dimension of each vertex in $\mathcal{V}_{\{i,j\}}$.

Then, apply the GAT algorithm on $v_m^i$ and $v_n^j$. Instead of shared linear transformation, we use two weight matrices, query weight $\mathbf{W}_q^m \in \mathbb{R}^{D_i^{'} \times D_i}$ and key weight $\mathbf{W}_k^n \in \mathbb{R}^{D_j^{'} \times D_j}$.

\begin{align}
    e^{m,n} &= a(\mathbf{W}_q^m v_m^i, \mathbf{W}_k^n v_n^j) \\
    \alpha^{m,n} &= \mathcal{S}_n(e^{m,n}) = \frac{\exp(e^{m,n})}{\sum_{l\in\mathcal{N}_n}\exp(e^{m,l})} 
\end{align}

After that, the final output feature for $v_m^i$ is computed. The value weight $\mathbf{W}_v^m \in \mathbb{R}^{D_i^{'} \times D_i}$ is applied to transform $v_n^j$:

\begin{equation}
\label{eq:vtx_sbg_0}
  \overline{v}_m^i = \parallel^L_{l=1} \sum_{n\in\mathcal{N}_m}\alpha^{m,n}_l\mathbf{W}_{v_l}^n v_n^j
\end{equation}

In the subgraph aspect, we assume that $\mathcal{N}_m$ includes all the vertices in subgraph $\mathcal{V}_j$. Current attention coefficient matrix is a vector $\mathcal{G}^m$, it can be regarded as a graph aggregated from $\mathcal{V}_j$ to $v_m^i$. The key, value weight for $\mathcal{V}_j$ is represented as $\mathcal{W}_{\{k, v\}} \in \mathbb{R}^{N_j \times D_j^{'} \times D_j}$. Then, the aggregation can be defined as follows:

\begin{align}
\label{eq:vtx_sbg_1}
  e^m &= a(\mathbf{W}_q^m v_m^i, \mathcal{W}_k \mathcal{V}_j), \quad \mathcal{G}^m = \mathcal{S}(e^m) \\
\label{eq:vtx_sbg_2}
  \overline{v}_m^i &= \parallel^L_{l=1} (\mathcal{G}^m_l \mathcal{W}_{v_l} \mathcal{V}_j)
\end{align}

Then, apply the algorithm defined by Equation \ref{eq:vtx_sbg_0}, \ref{eq:vtx_sbg_1}, \ref{eq:vtx_sbg_2} to all the vertices in $\mathcal{V}_i$. The aggregation form is now vertex set to vertex set, thus, we regard the vertex sets as subgraphs and vertex to vertex aggregation is transformed to subgraph aggregation. Also, the attention coefficient matrix $e$ is transformed as directional subgraph adjacency matrix $\mathcal{E}$. The query weight for $\mathcal{V}_i$ is represented as $\mathcal{W}_q$. 

\begin{align}
\label{eq:sbg_sbg0}
  \mathcal{E}^{i,j} &= a(\mathcal{W}_q \mathcal{V}_i, \mathcal{W}_k \mathcal{V}_j), \quad \mathcal{G}^{i, j} = \mathcal{S}(\mathcal{E}^{i,j}) \\
\label{eq:sbg_sbg1}
   \overline{\mathcal{V}}_i &= \parallel^L_{l=1}(\mathcal{G}_l^{i,j}\mathcal{W}_{v_l}\mathcal{V}_j)
\end{align}

Now the aggregation procedure is equal to multi-head cross-attention mechansim \cite{acl:mult}.

\textbf{Multimodal Subgraph Aggregation} Take $\mathcal{V}_i$ and $\mathcal{V}_j$, where $\{i,j\}\in\{t,v,a\}$ two modal sequences as an example, which is regarded as two vertex sets. Assuming that the unidirectional subgraph constructed by the two modal vertex sequences, so the adjacency matrix weight aggregation process of the corresponding subgraph is as follows.

\begin{equation}
  \mathcal{E}^{i,j} = (\mathcal{W}_q \mathcal{V}_j) \cdot (\mathcal{W}_k \mathcal{V}_i)^{\top}
\end{equation}

Then apply the softmax function.

% \vspace{-0.4cm} 
\begin{equation}
    \mathcal{G}^{i,j} &= \mathcal{S}(\mathcal{E}^{i,j})
\end{equation}

Finally, some of the edges in the subgraph are randomly masked which is realized by the dropout operation implemented on the adjacency matrix.

% \vspace{-0.4cm} 
\begin{equation}
  \mathcal{G}_{dropout}^{i,j} = \mathcal{D}(\mathcal{G}^{i,j})
\end{equation}

where $\mathcal{D}$ denotes the dropout function.

After the aggregation, fusion process is started, which is regarded as the directional information fusion procedure from $\mathcal{V}_j$ to $\mathcal{V}_i$.

\begin{equation}
  \overline{\mathcal{V}}_i = \mathcal{G}_{dropout}^{i,j} \mathcal{W}_v \mathcal{V}_j
\end{equation}

Then we extend the above operation globally as follows: 

\vspace{-0.3cm} 
\begin{align}
\label{eq:g_unstr}
  &\mathcal{G} = \mathcal{S} \circ \mathcal{D} (\mathcal{A}) \\
  &\overline{\mathcal{V}}_m = \mathcal{G} \mathcal{W}_v \mathcal{V}_m
\end{align} Where $\circ$ represents the function composition operation. Note: $\mathcal{A}$ is defined in Equation \ref{eq:adjmat_all}

Constructed graph structure in Equation \ref{eq:g_unstr} is actually unstructured at all, it loses sight of the separated modality-wise temporal features of the concatenated sequence which makes the sequence disordered. What is more, it over-fuses the inter-modal information, confuses inter-modal information and the intra-modal information and leaves way too much fine-grained information unconsidered. 

\begin{figure}[htpb]
\begin{center}
   \includegraphics[width=\columnwidth]{fig/example of mask.png}
   \caption{Example of to explain the necessity of interlaced mask.}
   \label{fig:example_mask}
% \vspace{-1.2cm}
\end{center}
\end{figure}

\subsection{Why the interlaced mask?}

\label{derivation:interlaced_mask}

Take the first block row in $\mathcal{A}$ as an example, which is $\mathbf{BR} = [\mathcal{E}^{t,t}, \mathcal{E}^{t,v}, \mathcal{E}^{t,a}]$. Knowing that $\mathcal{V}_m$ = $[\mathcal{V}_t;\mathcal{V}_v;\mathcal{V}_a]^{\top}$. Then the $\mathcal{E}^{t,t}$ is aggregated by $\mathcal{V}_t$ of $\mathcal{V}_m$ itself, $\mathcal{E}^{t,v}$ is aggregated by $\mathcal{V}_t$ and $\mathcal{V}_v$, $\mathcal{E}^{t,a}$ is aggregated by $\mathcal{V}_t$ and $\mathcal{V}_a$. And as defined in Equation \ref{eq:sbg_sbg0}, \ref{eq:sbg_sbg1}, the direction of aggregation of $\mathcal{E}^{i,j}$ is from $j$ to $i$. 

If the final output feature computation is performed without interlaced mask. It has to be noted that aggregation in this case is only performed on text modal $t$.

\begin{equation}
  \overline{\mathcal{V}}_t = \mathbf{BR} \cdot (\mathcal{W}_v \mathcal{V}_m)
\end{equation}

As shown in Figure \ref{fig:example_mask}. When we only mask one or two blocks (subgraphs), vertex sequences of different modals is considered to be the same sequence because they are spliced together. Thus making the temporal information disordered, which is absolutely not advisable.

\section{Algorithms}

\subsection{Interlaced Mask Generation Algorithm}
\label{apdx:crossmaskgen}

% \begin{breakablealgorithm}%[H]
%   \caption{Interlaced Mask Generation}
%   \label{alg:crossmask}
%   \textbf{Input}: Segmentation of the length of three-modal sequence $seg$ = $\{T_t,T_v,T_a\}$, Mode of the mask generation $mode$ $\in$ $\{inter, intra\}$, Direction of fusion procedure $dir$ $\in$ $\{forward, backward\}$; \\
%   \textbf{Output}: The generated mask of appointed mode and direction;
%   \begin{algorithmic}[1]
%     \STATE Let $\{l_t, l_v,l_a\} = seg$
%     \STATE Define segments $s1 = (0, l_t)$, $s2 = (l_t, l_t+l_v)$, $s3 = (l_t+l_v, l_t+l_v+l_a)$
%     \STATE Let $l_{sum} = l_t + l_v + l_a$
%     \STATE Initialize an empty list $\mathcal{M}_{list}$
%     \FOR{each $i$ in $[0, 1, 2]$}
%       \FOR{each element in $seg[i]$}
%         \STATE Initialize $m_{row}$ as a tensor of ones with size $l_{sum}$
%         \IF{$i == 0$}
%           \STATE Set $m_{row}[0:s1[1]] = 0$
%           \IF{$mode == inter$}
%             \IF{$dir == forward$}
%               \STATE Set $m_{row}[s3[0]:] = 0$
%             \ELSIF{$dir == backward$}
%               \STATE Set $m_{row}[s2[0]:s2[1]] = 0$
%             \ENDIF
%           \ENDIF
%         \ELSIF{$i == 1$}
%           \STATE Set $m_{row}[s2[0]:s2[1]] = 0$
%           \IF{$mode == inter$}
%             \IF{$dir == forward$}
%               \STATE Set $m_{row}[0:s1[1]] = 0$
%             \ELSIF{$dir == backward$}
%               \STATE Set $m_{row}[s3[0]:] = 0$
%             \ENDIF
%           \ENDIF
%         \ELSIF{$i == 2$}
%           \STATE Set $m_{row}[s3[0]:s3[1]] = 0$
%           \IF{$mode == inter$}
%             \IF{$dir == forward$}
%               \STATE Set $m_{row}[s2[0]:s2[1]] = 0$
%             \ELSIF{$dir == backward$}
%               \STATE Set $m_{row}[0:s1[1]] = 0$
%             \ENDIF
%           \ENDIF
%         \ENDIF
%         \STATE Append $m_{row}$ to $\mathcal{M}_{list}$
%       \ENDFOR
%     \ENDFOR
%     \IF{$mode == inter$}
%       \STATE Let $\mathcal{M} = \text{Stack}(\mathcal{M}_{list})$
%       \STATE \textbf{return} $\text{GenerateMask}(\mathcal{M})$
%     \ELSIF{$mode == intra$}
%       \STATE \textbf{return} $\text{GenerateMask}(|\text{Stack}(\mathcal{M}_{list}) - 1)|)$
%     \ENDIF
%   \end{algorithmic}
% \end{breakablealgorithm}

\subsection{Extended Long Short Term Memory with Matrix Memory}
\label{apdx:xlstm}

\begin{figure}[htpb]
\begin{center}
   \includegraphics[width=\columnwidth]{fig/mLSTM.png}
    \caption{Parallelized Extended LSTM with Matrix Memory.}
    \label{fig:xlstm}
% \vspace{-1.2cm}
\end{center}
\end{figure}

\begin{align}  
  C_t &= f_t C_{t-1} + i_t v_t k_t^{\top} \\
  n_t &= f_t n_{t-1} + i_t k_t \\
  h_t &= o_t \odot \tilde{h}_t, 
  &\tilde{h}_t &= \frac{C_t q_t}{\text{max}\{|n_t^{\top}q_t|, 1\}} \\
  q_t &= W_q x_t + b_q \\
  k_t &= \frac{1}{\sqrt{d}}W_k x_t + b_k \\
  v_t &= W_v x_t + b_v \\
  i_t &= \exp(\tilde{i}_t), 
  &\tilde{i}_t &= w_i^{\top}x_t + b_i \\
  f_t &= \sigma{(\tilde{f}_t)} \text{OR} \exp{(\tilde{f}_t)}, 
  &\tilde{f}_t &= w_f^{\top} x_t + b_f \\
  o_t &= \sigma{(\tilde{o}_t)}, 
  &\tilde{o}_t &= W_o x_t + b_o
\end{align}

The forward pass of mLSTM can be described as the above equation group, while the detailed architecture is shown in Figure \ref{fig:xlstm}


The detailed generation method of cross mask for not only the forward and backward inter-fusion but also the intra-enhancement is shown on the algorithm table above. It is of vital importance for our model to accurately construct the graph structure of the concatenated sequence list. 

Also, the masks could be constructed during the initialization procedure.

\end{document}
"""

import re

pattern = r'^\$.*\$$'
matches = re.findall(pattern, text)

# 计算匹配项的数量
count = len(matches)

print(f"匹配项的数量是：{count}")