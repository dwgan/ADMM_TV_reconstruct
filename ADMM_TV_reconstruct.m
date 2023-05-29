function xp=ADMM_TV_reconstruct(A,b,delta,lambda,iteratMax)
    [~,N]=size(A);
    [Dh,Dv]=TVOperatorGen(sqrt(N));
    D=sparse([(Dh)',(Dv)']');
    d=D*ones(N,1);
    p=ones(2*N,1)/delta;
    invDD=inv(A'*A+delta*(D'*D));
    for ii=1:iteratMax
        x=invDD*(A'*b+delta*D'*(d-p));
        d=wthresh(D*x+p,'s',lambda/delta);
        p=p+D*x-d;
    end
    xp=x;
end