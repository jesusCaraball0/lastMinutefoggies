"""boosted_v1: own action-observation history plus explicitly stale sightings.

Uses only the player's views and previously executed own moves. Current target
moves and the referee's hidden board never enter the feature representation.
"""
from collections import deque
import chess
import torch
import torch.nn.functional as F

SYMBOLS = 'PNBRQKpnbrqk?-'
UNKNOWN = SYMBOLS.index('?')
TURNS = 20
RECORD_SIZE = 73
AUX_SIZE = 130  # current castling bits, last seen64, ages64
ROW_SIZE = RECORD_SIZE + AUX_SIZE
TOKEN_COUNT = 2 + TURNS * RECORD_SIZE + AUX_SIZE
CHANNELS = TURNS * 90 + 19  # rights2, last-seen14, age1, seen1, visible1
MAGIC = (255, 1)
DIRECTIONS = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
KNIGHTS = [(1,2),(2,1),(2,-1),(1,-2),(-1,-2),(-2,-1),(-2,1),(-1,2)]


def move_code(move, side):
    """73 AlphaZero-style move classes, anchored at the from square.

    Rays56 + knights8 + underpromotions9. Queen promotions use a ray class.
    Board coordinates stay absolute; underpromotion left/right is player-relative.
    """
    dx = chess.square_file(move.to_square)-chess.square_file(move.from_square)
    dy = chess.square_rank(move.to_square)-chess.square_rank(move.from_square)
    if move.promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK):
        return 64 + (chess.KNIGHT,chess.BISHOP,chess.ROOK).index(move.promotion)*3 + dx*(1 if side else -1)+1
    if (dx,dy) in KNIGHTS:
        return 56 + KNIGHTS.index((dx,dy))
    distance = max(abs(dx),abs(dy))
    if dx and dy and abs(dx)!=abs(dy):
        raise ValueError('Move has no ray/knight encoding')
    direction = (dx//distance,dy//distance) if distance else (0,0)
    if direction not in DIRECTIONS or not 1 <= distance <= 7:
        raise ValueError('Move has no ray/knight encoding')
    return DIRECTIONS.index(direction)*7 + distance-1


class BoostedHistory:
    def __init__(self, side):
        self.side = bool(side)
        self.records = deque(maxlen=TURNS)
        self.current = None
        self.last_seen = [UNKNOWN]*64
        self.last_visible = [None]*64
        self.ply = None
        self.previous_move = (64,73,64)
        self.own_capture = self.enemy_capture = 64

    def observe(self, view, ply):
        if len(view)!=64 or not set(view)<=set(SYMBOLS):
            raise ValueError('Expected a 64-square fog observation')
        if self.ply is not None and ply!=self.ply+1:
            raise ValueError('Boosted history requires every consecutive player observation')
        ours = 'PNBRQK' if self.side else 'pnbrqk'
        # Detect losses only after an opponent turn, not after our own move.
        if self.current is not None and ((ply-1)%2==0)!=self.side:
            lost=[i for i,c in enumerate(self.current) if c in ours and view[i]!=c]
            if len(lost)>1:raise ValueError('More than one own piece disappeared on an opponent turn')
            self.enemy_capture=lost[0] if lost else 64
        for i,c in enumerate(view):
            if c!='?':
                self.last_seen[i]=SYMBOLS.index(c);self.last_visible[i]=ply
        self.current,self.ply=view,ply
        if (ply%2==0)==self.side:
            self.records.appendleft([SYMBOLS.index(c) for c in view]+list(self.previous_move)+[
                self.own_capture,self.enemy_capture,int(self.side),0,0,1])

    def record_move(self, move):
        move=chess.Move.from_uci(move) if isinstance(move,str) else move
        if self.current is None:raise ValueError('Move recorded before any observation')
        ours='PNBRQK' if self.side else 'pnbrqk';enemy='pnbrqk' if self.side else 'PNBRQK'
        if self.current[move.from_square] not in ours:raise ValueError('Own move must originate on an observed own piece')
        self.previous_move=(move.from_square,move_code(move,self.side),move.to_square)
        self.own_capture=move.to_square if self.current[move.to_square] in enemy else 64
        # En-passant feedback can be derived from our actual move and visible pieces.
        if self.current[move.from_square].lower()=='p' and chess.square_file(move.from_square)!=chess.square_file(move.to_square) and self.current[move.to_square]=='-':
            victim=chess.square(chess.square_file(move.to_square),chess.square_rank(move.from_square))
            if self.current[victim].lower()=='p' and self.current[victim] in enemy:self.own_capture=victim

    def aux(self, rights):
        rank=0 if self.side else 7
        bits=[int(bool(rights & chess.BB_SQUARES[chess.square(file,rank)])) for file in (7,0)]
        ages=[255 if t is None else min(254,self.ply-t) for t in self.last_visible]
        return bits+self.last_seen+ages

    def compact_row(self, rights):
        if not self.records:raise ValueError('No own-turn observation available')
        return self.records[0]+self.aux(rights)

    def encoded_tokens(self, rights):
        records=list(self.records)+[[0]*RECORD_SIZE]*(TURNS-len(self.records))
        return list(MAGIC)+[v for row in records for v in row]+self.aux(rights)


def boosted_planes(tokens, device='cpu'):
    if tokens.shape[-1]!=TOKEN_COUNT:raise ValueError('Invalid boosted_v1 token width')
    leading=tokens.shape[:-1]
    flat=tokens.to(device=device,dtype=torch.long).reshape(-1,TOKEN_COUNT)
    records=flat[:,2:2+TURNS*RECORD_SIZE].reshape(-1,TURNS,RECORD_SIZE)
    valid=records[...,72].float()
    board=F.one_hot(records[...,:64],14).transpose(-1,-2).float()*valid[...,None,None]
    origin,code=records[...,64],records[...,65]
    move_valid=(origin<64)&(code<73)&(valid>0)
    move_planes=torch.zeros(len(flat),TURNS,73*64,device=device)
    move_planes.scatter_(-1,(code*64+origin).clamp(max=73*64-1)[...,None],move_valid.float()[...,None])
    move_planes=move_planes.reshape(-1,TURNS,73,64)
    captures=F.one_hot(records[...,67:69],65)[...,:64].float()*valid[...,None,None]
    color=(records[...,69].float()*valid)[...,None,None].expand(-1,-1,1,64)
    history=torch.cat((board,move_planes,captures,color),dim=2).reshape(-1,TURNS*90,64)
    aux=flat[:,2+TURNS*RECORD_SIZE:]
    rights=aux[:,:2].float()[...,None].expand(-1,-1,64)
    last=aux[:,2:66]
    last_planes=F.one_hot(last,14).transpose(-1,-2).float()
    age=aux[:,66:130].clamp(max=254).float()[:,None]/254
    seen=(last!=UNKNOWN).float()[:,None]
    visible=(records[:,0,:64]!=UNKNOWN).float()[:,None]
    planes=torch.cat((history,rights,last_planes,age,seen,visible),dim=1)
    return planes.reshape(*leading,CHANNELS,8,8)
